using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace RealTimeLhzScripts
{
    /// <summary>
    /// One complete real-time endpoint for one avatar.
    ///
    /// The endpoint owns its TCP listener, codec-facing text parser, latency
    /// feedback, face driver and Mixamo/lhz body driver. Duplicate this same
    /// component for another avatar and change only the Inspector references
    /// and ports. Every endpoint must use a unique data port.
    /// </summary>
    public partial class RealTimeLhzAvatarEndpoint : MonoBehaviour
    {
        private const int FaceChannelCount = 52;
        private const int BodyChannelCount = 99;
        private const int ExpectedFieldCount = 1 + FaceChannelCount + BodyChannelCount;

        [Header("Network")]
        [SerializeField] private string listenAddress = "127.0.0.1";
        [SerializeField] private int listenPort = 8890;
        [SerializeField] private string feedbackAddress = "127.0.0.1";
        [SerializeField] private int feedbackPort = 9890;
        [Tooltip("Receiver (Unity/Windows) clock minus sender clock, in milliseconds. " +
                 "Positive when the receiver clock is ahead. grpc2socket subtracts this from network/total latency.")]
        [SerializeField] private float clockOffsetMs = 0f;
        [SerializeField] private int receiveBufferSize = 8192;
        [SerializeField] private int maxFrameBytes = 65536;
        [Tooltip("Maximum decoded frames waiting for the Unity main thread. Oldest frames are dropped when this is exceeded.")]
        [SerializeField] private int maxQueuedFrames = 240;
        [Tooltip("Maximum timing feedback messages waiting for LateUpdate. Oldest messages are dropped when this is exceeded.")]
        [SerializeField] private int maxPendingFeedbacks = 240;

        [Header("Avatar")]
        [SerializeField] private GameObject avatarRoot;
        [SerializeField] private SkinnedMeshRenderer faceRenderer;
        [SerializeField] private bool enableAxisCorrection = true;
        [SerializeField] private float poseOffsetX;
        [SerializeField] private float poseOffsetY;
        [SerializeField] private float poseOffsetZ;

        [Header("Grounding and Gravity")]
        [Tooltip("Use a CharacterController with real gravity and ground collision instead of following the camera hip Y.")]
        [SerializeField] private bool enableGrounding = true;
        [Tooltip("Space triggers the same physical jump as the old grounding component.")]
        [SerializeField] private bool enableSpaceJump = true;
        [SerializeField] private LiveAvatarGrounding liveGrounding;

        private sealed class ReceivedFrame
        {
            public string Payload;
            public long TransitTimestampMs;
        }

        private sealed class PendingFeedback
        {
            public long BeginTimestampMs;
            public long TransitTimestampMs;
        }

        private readonly Queue<ReceivedFrame> receivedFrames = new Queue<ReceivedFrame>();
        private readonly object receivedFramesLock = new object();
        private readonly Queue<PendingFeedback> pendingFeedbacks = new Queue<PendingFeedback>();
        private readonly object pendingFeedbacksLock = new object();
        private int droppedReceivedFrameCount;
        private int droppedFeedbackCount;

        private TcpListener tcpListener;
        private TcpClient activeClient;
        private Thread listenerThread;
        private volatile bool receiverRunning;

        private void Awake()
        {
            InitializeFaceDriver();
            InitializeBodyDriver();
        }

        private void OnEnable()
        {
            StartReceiver();
        }

        private void OnDisable()
        {
            StopReceiver();
            ResetLiveBodyTracking();
        }

        private void OnDestroy()
        {
            StopReceiver();
            DestroyAxisCorrectionRoot();
        }

        private void Update()
        {
            while (true)
            {
                ReceivedFrame frame;
                lock (receivedFramesLock)
                {
                    if (receivedFrames.Count == 0) break;
                    frame = receivedFrames.Dequeue();
                }

                ProcessReceivedFrame(frame.Payload, frame.TransitTimestampMs);
            }

            if (enableSpaceJump && liveGrounding != null && Input.GetKeyDown(KeyCode.Space))
                liveGrounding.RequestJump();
        }

        private void LateUpdate()
        {
            while (true)
            {
                PendingFeedback feedback;
                lock (pendingFeedbacksLock)
                {
                    if (pendingFeedbacks.Count == 0) break;
                    feedback = pendingFeedbacks.Dequeue();
                }

                long finalTimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                SendTimingFeedback(
                    feedback.BeginTimestampMs,
                    feedback.TransitTimestampMs,
                    finalTimestampMs);
            }
        }

        private void StartReceiver()
        {
            if (receiverRunning) return;
            if (listenPort < 1 || listenPort > 65535)
            {
                Debug.LogError($"[RealTimeLhzAvatarEndpoint] Invalid listen port: {listenPort}");
                return;
            }

            receiverRunning = true;
            listenerThread = new Thread(ListenLoop)
            {
                IsBackground = true,
                Name = $"RealTimeLhzAvatarEndpoint:{listenPort}"
            };
            listenerThread.Start();
        }

        private void StopReceiver()
        {
            receiverRunning = false;

            try { activeClient?.Close(); } catch { }
            activeClient = null;

            try { tcpListener?.Stop(); } catch { }
            tcpListener = null;

            if (listenerThread != null && listenerThread != Thread.CurrentThread)
            {
                try { listenerThread.Join(2000); } catch { }
            }
            listenerThread = null;

            lock (receivedFramesLock) receivedFrames.Clear();
            lock (pendingFeedbacksLock) pendingFeedbacks.Clear();
        }

        private void ListenLoop()
        {
            try
            {
                IPAddress address = ResolveListenAddress(listenAddress);
                tcpListener = new TcpListener(address, listenPort);
                tcpListener.Start();
                Debug.Log($"[RealTimeLhzAvatarEndpoint] Listening on {listenAddress}:{listenPort}");

                while (receiverRunning)
                {
                    TcpClient client;
                    try
                    {
                        client = tcpListener.AcceptTcpClient();
                    }
                    catch (SocketException)
                    {
                        if (!receiverRunning) break;
                        continue;
                    }

                    activeClient = client;
                    HandleClient(client);
                    activeClient = null;
                }
            }
            catch (Exception exception)
            {
                if (receiverRunning)
                    Debug.LogError($"[RealTimeLhzAvatarEndpoint] Listener failed: {exception.Message}");
            }
            finally
            {
                try { tcpListener?.Stop(); } catch { }
                tcpListener = null;
            }
        }

        private void HandleClient(TcpClient client)
        {
            try
            {
                using (client)
                using (NetworkStream stream = client.GetStream())
                {
                    byte[] readBuffer = new byte[Mathf.Max(1024, receiveBufferSize)];
                    int frameCapacity = Mathf.Clamp(maxFrameBytes, 1024, 4 * 1024 * 1024);
                    List<byte> frameBytes = new List<byte>(frameCapacity);

                    while (receiverRunning)
                    {
                        int bytesRead;
                        try
                        {
                            bytesRead = stream.Read(readBuffer, 0, readBuffer.Length);
                        }
                        catch (IOException)
                        {
                            break;
                        }
                        catch (SocketException)
                        {
                            break;
                        }

                        if (bytesRead <= 0) break;

                        for (int i = 0; i < bytesRead; i++)
                        {
                            byte value = readBuffer[i];
                            if (value == (byte)'\n')
                            {
                                if (frameBytes.Count > 0)
                                {
                                    long transitTimestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                                    EnqueueReceivedFrame(frameBytes, transitTimestampMs);
                                }
                                frameBytes.Clear();
                                continue;
                            }

                            if (frameBytes.Count < frameCapacity)
                            {
                                if (value != (byte)'\r') frameBytes.Add(value);
                            }
                            else
                            {
                                frameBytes.Clear();
                                Debug.LogWarning("[RealTimeLhzAvatarEndpoint] Discarded an oversized frame.");
                            }
                        }
                    }
                }
            }
            catch (Exception exception)
            {
                if (receiverRunning)
                    Debug.LogWarning($"[RealTimeLhzAvatarEndpoint] Client error: {exception.Message}");
            }
        }

        private void EnqueueReceivedFrame(List<byte> frameBytes, long transitTimestampMs)
        {
            string payload = Encoding.UTF8.GetString(frameBytes.ToArray());
            bool dropped = false;
            lock (receivedFramesLock)
            {
                int queueLimit = Mathf.Max(1, maxQueuedFrames);
                while (receivedFrames.Count >= queueLimit)
                {
                    receivedFrames.Dequeue();
                    dropped = true;
                }
                receivedFrames.Enqueue(new ReceivedFrame
                {
                    Payload = payload,
                    TransitTimestampMs = transitTimestampMs
                });
            }

            if (dropped && Interlocked.Increment(ref droppedReceivedFrameCount) % 30 == 1)
            {
                Debug.LogWarning(
                    $"[RealTimeLhzAvatarEndpoint] Receive queue is overloaded; " +
                    $"dropped {droppedReceivedFrameCount} oldest frame(s).");
            }
        }

        private void ProcessReceivedFrame(string payload, long transitTimestampMs)
        {
            if (string.IsNullOrWhiteSpace(payload)) return;

            string[] fields = payload.Split(',');
            if (fields.Length != ExpectedFieldCount)
            {
                Debug.LogWarning(
                    $"[RealTimeLhzAvatarEndpoint] Discarded frame with {fields.Length} fields; " +
                    $"expected {ExpectedFieldCount}.");
                return;
            }

            if (!long.TryParse(fields[0], NumberStyles.Integer, CultureInfo.InvariantCulture,
                out long beginTimestampMs))
            {
                Debug.LogWarning("[RealTimeLhzAvatarEndpoint] Discarded frame with an invalid timestamp.");
                return;
            }

            float[] faceData = new float[FaceChannelCount];
            float[] bodyData = new float[BodyChannelCount];
            if (!TryParseFiniteValues(fields, 1, faceData) ||
                !TryParseFiniteValues(fields, 1 + FaceChannelCount, bodyData))
            {
                Debug.LogWarning("[RealTimeLhzAvatarEndpoint] Discarded frame with invalid numeric data.");
                return;
            }

            ApplyFaceData(faceData);
            ApplyBodyData(bodyData, beginTimestampMs);

            bool droppedFeedback = false;
            lock (pendingFeedbacksLock)
            {
                int queueLimit = Mathf.Max(1, maxPendingFeedbacks);
                while (pendingFeedbacks.Count >= queueLimit)
                {
                    pendingFeedbacks.Dequeue();
                    droppedFeedback = true;
                }
                pendingFeedbacks.Enqueue(new PendingFeedback
                {
                    BeginTimestampMs = beginTimestampMs,
                    TransitTimestampMs = transitTimestampMs
                });
            }

            if (droppedFeedback && Interlocked.Increment(ref droppedFeedbackCount) % 30 == 1)
            {
                Debug.LogWarning(
                    $"[RealTimeLhzAvatarEndpoint] Feedback queue is overloaded; " +
                    $"dropped {droppedFeedbackCount} oldest feedback message(s).");
            }
        }

        private static bool TryParseFiniteValues(string[] fields, int startIndex, float[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                if (!float.TryParse(fields[startIndex + i], NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float value) ||
                    float.IsNaN(value) || float.IsInfinity(value))
                    return false;
                values[i] = value;
            }
            return true;
        }

        private void SendTimingFeedback(long beginTimestampMs, long transitTimestampMs, long finalTimestampMs)
        {
            try
            {
                string message = string.Format(
                    CultureInfo.InvariantCulture,
                    "timing:{0},{1},{2},{3}",
                    beginTimestampMs,
                    transitTimestampMs,
                    finalTimestampMs,
                    clockOffsetMs);
                using (TcpClient client = new TcpClient())
                {
                    client.Connect(feedbackAddress, feedbackPort);
                    using (NetworkStream stream = client.GetStream())
                    {
                        byte[] bytes = Encoding.UTF8.GetBytes(message);
                        stream.Write(bytes, 0, bytes.Length);
                    }
                }
            }
            catch (Exception exception)
            {
                Debug.LogWarning($"[RealTimeLhzAvatarEndpoint] Timing feedback failed: {exception.Message}");
            }
        }

        private static IPAddress ResolveListenAddress(string address)
        {
            if (string.IsNullOrWhiteSpace(address) || address == "0.0.0.0")
                return IPAddress.Any;
            return IPAddress.Parse(address);
        }
    }

    public partial class RealTimeLhzAvatarEndpoint
    {
        private const int BlendShapeCount = 52;
        private readonly Dictionary<int, string> blendShapeNamesByChannel = new Dictionary<int, string>();
        private readonly Dictionary<string, int> blendShapeIndicesByName = new Dictionary<string, int>();
        private bool faceInitialized;

        private void InitializeFaceDriver()
        {
            if (faceRenderer == null && avatarRoot != null)
            {
                SkinnedMeshRenderer[] renderers = avatarRoot.GetComponentsInChildren<SkinnedMeshRenderer>(true);
                foreach (SkinnedMeshRenderer renderer in renderers)
                {
                    if (renderer.sharedMesh != null && renderer.sharedMesh.blendShapeCount > 0)
                    {
                        faceRenderer = renderer;
                        break;
                    }
                }
            }

            InitializeBlendShapeChannelMap();
            if (faceRenderer == null || faceRenderer.sharedMesh == null ||
                faceRenderer.sharedMesh.blendShapeCount == 0)
            {
                Debug.LogWarning("[RealTimeLhzAvatarEndpoint] No BlendShape renderer is assigned.");
                return;
            }

            Dictionary<string, int> modelNames = new Dictionary<string, int>();
            for (int i = 0; i < faceRenderer.sharedMesh.blendShapeCount; i++)
                modelNames[faceRenderer.sharedMesh.GetBlendShapeName(i)] = i;

            string commonPrefix = DetectCommonBlendShapePrefix(modelNames.Keys);
            foreach (KeyValuePair<int, string> channel in blendShapeNamesByChannel)
            {
                string standardName = channel.Value;
                int modelIndex;
                if (modelNames.TryGetValue(commonPrefix + standardName, out modelIndex) ||
                    modelNames.TryGetValue(standardName, out modelIndex))
                {
                    blendShapeIndicesByName[standardName] = modelIndex;
                    continue;
                }

                foreach (KeyValuePair<string, int> modelName in modelNames)
                {
                    if (modelName.Key.EndsWith(standardName, StringComparison.Ordinal))
                    {
                        blendShapeIndicesByName[standardName] = modelName.Value;
                        break;
                    }
                }
            }

            faceInitialized = true;
        }

        private void InitializeBlendShapeChannelMap()
        {
            string[] names =
            {
                "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
                "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
                "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
                "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
                "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
                "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
                "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
                "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
                "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
                "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
                "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
                "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
                "noseSneerLeft", "noseSneerRight"
            };

            for (int i = 0; i < names.Length; i++)
                blendShapeNamesByChannel[i] = names[i];
        }

        private static string DetectCommonBlendShapePrefix(IEnumerable<string> modelNames)
        {
            foreach (string modelName in modelNames)
            {
                if (modelName.EndsWith("browDownLeft", StringComparison.Ordinal) &&
                    modelName.Length > "browDownLeft".Length)
                    return modelName.Substring(0, modelName.Length - "browDownLeft".Length);
            }
            return string.Empty;
        }

        private void ApplyFaceData(float[] faceData)
        {
            if (!faceInitialized || faceRenderer == null || faceRenderer.sharedMesh == null) return;

            for (int channel = 0; channel < Mathf.Min(BlendShapeCount, faceData.Length); channel++)
            {
                string standardName;
                if (!blendShapeNamesByChannel.TryGetValue(channel, out standardName)) continue;

                int modelIndex;
                if (!blendShapeIndicesByName.TryGetValue(standardName, out modelIndex)) continue;
                float weight = Mathf.Clamp(faceData[channel] * 100f, 0f, 100f);
                faceRenderer.SetBlendShapeWeight(modelIndex, weight);
            }
        }
    }


    public partial class RealTimeLhzAvatarEndpoint
    {
        private GameObject bodyAxisCorrectionRoot;
        private Transform originalAvatarParent;
        private Vector3 originalAvatarLocalPosition;
        private Quaternion originalAvatarLocalRotation;
        private Transform[] boneTransforms;
        private bool bodyInitialized = false;
        private bool bodyInitWarningLogged = false;

        // Math & Orientation Variables
        private Vector3 bodyForward, leftHandNormal, rightHandNormal, headGaze;
        private Vector3 restBodyForward; // saved at rest to detect bodyForward flips
        private Quaternion[] restLocalRotations;
        private bool hasBodyPose;
        private bool livePoseAnchored;
        private Vector3 liveInputOrigin, liveSceneOrigin, lastLiveBodyForward;
        private float lastLiveFrameTime;
        private long lastLiveTimestamp;
        private float[] lastRawBodyData;
        private float[] lastScaledBodyData;
        private long lastBodyTimestamp;
        private Vector3 leftUpperLegPosition, rightUpperLegPosition, leftThumbPosition, leftIndexPosition, rightThumbPosition, rightIndexPosition, leftLittlePosition, rightLittlePosition, headPosition;
        private Vector3 leftUpperLegDirection, rightUpperLegDirection, leftLowerLegDirection, rightLowerLegDirection, leftFootDirection, rightFootDirection, spineDirection, chestDirection, upperChestDirection, leftShoulderDirection, leftUpperArmDirection, leftLowerArmDirection, rightShoulderDirection, rightUpperArmDirection, rightLowerArmDirection, leftThumbDirection, leftIndexDirection, leftLittleDirection, rightThumbDirection, rightIndexDirection, rightLittleDirection;
        private Quaternion hipsCalibration, leftUpperLegCalibration, rightUpperLegCalibration, leftLowerLegCalibration, rightLowerLegCalibration, leftFootCalibration, rightFootCalibration, spineCalibration, chestCalibration, upperChestCalibration, leftShoulderCalibration, rightShoulderCalibration, leftUpperArmCalibration, rightUpperArmCalibration, leftLowerArmCalibration, rightLowerArmCalibration, leftHandCalibration, rightHandCalibration, leftThumbCalibration, leftIndexCalibration, leftLittleCalibration, rightThumbCalibration, rightIndexCalibration, rightLittleCalibration, neckCalibration;

        // Per-bone debug capture (for AvatarExporter.CaptureState). Indexed by boneTransforms index.
        private Quaternion[] debugRestRotations = new Quaternion[38];
        private Vector3[] debugAnimationDirections = new Vector3[38];
        private Vector3[] debugUpDirections = new Vector3[38];

        public Quaternion[] DebugRestRotations => debugRestRotations;
        public Vector3[] DebugAnimationDirections => debugAnimationDirections;
        public Vector3[] DebugUpDirections => debugUpDirections;

        private void InitializeBodyDriver()
        {
            if (avatarRoot == null)
            {
                Debug.LogError("[RealTimeLhzAvatarEndpoint] avatarRoot is not assigned.");
                bodyInitialized = false;
                return;
            }

            boneTransforms = new Transform[38];

            // Helper to find bones by Mixamo name
            Transform FindBone(string name)
            {
                Transform t = FindBodyBoneRecursive(avatarRoot.transform, "mixamorig:" + name);
                if (t == null) Debug.LogWarning($"Bone mixamorig:{name} not found on {avatarRoot.name}!");
                return t;
            }

            // Map Mixamo Hierarchy to your Index System
            boneTransforms[0] = FindBone("Hips");
            boneTransforms[1] = FindBone("Spine");
            boneTransforms[2] = FindBone("Spine1");
            boneTransforms[3] = FindBone("Spine2");
            boneTransforms[4] = FindBone("LeftShoulder");
            boneTransforms[5] = FindBone("LeftArm");
            boneTransforms[6] = FindBone("LeftForeArm");
            boneTransforms[7] = FindBone("LeftHand");
            boneTransforms[8] = FindBone("LeftHandThumb2");
            boneTransforms[9] = FindBone("LeftHandIndex2") ?? boneTransforms[8];
            boneTransforms[10] = FindBone("LeftHandPinky2") ?? boneTransforms[8];
            boneTransforms[11] = FindBone("RightShoulder");
            boneTransforms[12] = FindBone("RightArm");
            boneTransforms[13] = FindBone("RightForeArm");
            boneTransforms[14] = FindBone("RightHand");
            boneTransforms[15] = FindBone("RightHandThumb2");
            boneTransforms[16] = FindBone("RightHandIndex2") ?? boneTransforms[15];
            boneTransforms[17] = FindBone("RightHandPinky2") ?? boneTransforms[15];
            boneTransforms[18] = FindBone("LeftUpLeg");
            boneTransforms[19] = FindBone("LeftLeg");
            boneTransforms[20] = FindBone("LeftFoot");
            boneTransforms[21] = FindBone("RightUpLeg");
            boneTransforms[22] = FindBone("RightLeg");
            boneTransforms[23] = FindBone("RightFoot");
            boneTransforms[24] = FindBone("Neck");
            boneTransforms[25] = FindBone("LeftToeBase");
            boneTransforms[26] = FindBone("RightToeBase");
            boneTransforms[27] = FindBone("LeftHandThumb3");
            boneTransforms[28] = FindBone("RightHandThumb3");
            boneTransforms[29] = FindBone("LeftHandIndex3") ?? boneTransforms[8];
            boneTransforms[30] = FindBone("RightHandIndex3") ?? boneTransforms[15];
            boneTransforms[31] = FindBone("LeftHandPinky3") ?? boneTransforms[8];
            boneTransforms[32] = FindBone("RightHandPinky3") ?? boneTransforms[15];
            boneTransforms[33] = FindBone("LeftHandMiddle2") ?? boneTransforms[8];
            boneTransforms[34] = FindBone("RightHandMiddle2") ?? boneTransforms[15];
            boneTransforms[35] = FindBone("LeftHandRing2") ?? boneTransforms[8];
            boneTransforms[36] = FindBone("RightHandRing2") ?? boneTransforms[15];
            boneTransforms[37] = FindBone("Head");

            // Hips through neck plus both toe bases are required by the
            // calibration matrices. Finger tips may use lhz's fallbacks.
            for (int i = 0; i <= 26; i++)
            {
                if (boneTransforms[i] == null)
                {
                    Debug.LogError($"[RealTimeLhzAvatarEndpoint] Critical bone {i} missing — controller disabled.");
                    bodyInitialized = false;
                    return;
                }
            }

            InitializeBodyCalibration();
            CacheBodyRestPose();
            SetupBodyAxisCorrection();
            bodyInitialized = true;
        }

        // --- PUBLIC METHOD FOR DATA LOADER ---

        private Quaternion SafeLookRotation(Vector3 bodyForward)
        {
            if (bodyForward.sqrMagnitude < 0.0001f) return Quaternion.identity;
            return Quaternion.LookRotation(bodyForward);
        }

        private Quaternion SafeLookRotation(Vector3 bodyForward, Vector3 up)
        {
            if (bodyForward.sqrMagnitude < 0.0001f) return Quaternion.identity;
            if (up.sqrMagnitude < 0.0001f) return Quaternion.LookRotation(bodyForward.normalized);
            Vector3 nForward = bodyForward.normalized;
            Vector3 nUp = up.normalized;
            if (Mathf.Abs(Vector3.Dot(nForward, nUp)) > 0.999f)
                nUp = Mathf.Abs(Vector3.Dot(nForward, Vector3.up)) > 0.999f ? Vector3.right : Vector3.up;
            return Quaternion.LookRotation(nForward, nUp);
        }

        private Vector3 SafeTriangleNormal(Vector3 a, Vector3 b, Vector3 c)
        {
            Vector3 d1 = a - b;
            Vector3 d2 = a - c;
            Vector3 cross = Vector3.Cross(d1, d2);
            if (cross.sqrMagnitude < 0.0001f) return Vector3.up;
            return cross.normalized;
        }

        /// <summary>
        /// When finger bones are missing (28-bone SkinTokens rig), fallback bones
        /// point to the same thumb Transform. This creates zero-length direction
        /// vectors. We can't change the bone Transforms, but we record synthetic
        /// rest positions for Mid matrix computation.
        /// </summary>
        private Vector3[] boneRestPositions;
        private bool usingSyntheticPositions;

        private void EnsureDistinctFingerBones()
        {
            boneRestPositions = new Vector3[38];
            for (int i = 0; i < 38; i++)
                if (boneTransforms[i] != null)
                    boneRestPositions[i] = boneTransforms[i].position;

            // If LeftHandIndex2 (9) == LeftHandThumb2 (8), create synthetic offset
            if (boneTransforms[9] == boneTransforms[8] && boneTransforms[7] != null)
            {
                Vector3 handForward = boneTransforms[7].forward * 0.03f;
                Vector3 handRight = boneTransforms[7].right * 0.02f;
                boneRestPositions[9] = boneTransforms[7].position + handForward + handRight;   // index
                boneRestPositions[29] = boneRestPositions[9] + handForward * 0.5f;             // index tip
                boneRestPositions[10] = boneTransforms[7].position + handForward - handRight;   // pinky
                boneRestPositions[31] = boneRestPositions[10] + handForward * 0.5f;            // pinky tip
                boneRestPositions[33] = boneRestPositions[9];                                   // middle
                boneRestPositions[35] = boneRestPositions[10];                                  // ring
                usingSyntheticPositions = true;
            }

            // Same for right hand
            if (boneTransforms[16] == boneTransforms[15] && boneTransforms[14] != null)
            {
                Vector3 handForward = boneTransforms[14].forward * 0.03f;
                Vector3 handRight = boneTransforms[14].right * 0.02f;
                boneRestPositions[16] = boneTransforms[14].position + handForward + handRight;
                boneRestPositions[30] = boneRestPositions[16] + handForward * 0.5f;
                boneRestPositions[17] = boneTransforms[14].position + handForward - handRight;
                boneRestPositions[32] = boneRestPositions[17] + handForward * 0.5f;
                boneRestPositions[34] = boneRestPositions[16];
                boneRestPositions[36] = boneRestPositions[17];
                usingSyntheticPositions = true;
            }
        }

        private bool IsAliasedFinger(int idx)
        {
            if (idx < 0 || idx >= boneTransforms.Length || boneTransforms[idx] == null) return true;

            bool leftAlias = idx != 8 && boneTransforms[8] != null && boneTransforms[idx] == boneTransforms[8] &&
                (idx == 9 || idx == 10 || idx == 29 || idx == 31 || idx == 33 || idx == 35);
            bool rightAlias = idx != 15 && boneTransforms[15] != null && boneTransforms[idx] == boneTransforms[15] &&
                (idx == 16 || idx == 17 || idx == 30 || idx == 32 || idx == 34 || idx == 36);

            return leftAlias || rightAlias;
        }

        private void SetBone(int idx, Vector3 dir, Vector3 up, Quaternion mid)
        {
            if (IsAliasedFinger(idx)) return;
            if (dir.sqrMagnitude < 0.0001f) return;
            RecordDebugDir(idx, dir, up);
            boneTransforms[idx].rotation = SafeLookRotation(dir, up) * Quaternion.Inverse(mid);
        }

        private void SetBone(int idx, Vector3 dir, Quaternion mid)
        {
            if (IsAliasedFinger(idx)) return;
            if (dir.sqrMagnitude < 0.0001f) return;
            RecordDebugDir(idx, dir, Vector3.zero);
            boneTransforms[idx].rotation = SafeLookRotation(dir) * Quaternion.Inverse(mid);
        }

        private void RecordDebugDir(int idx, Vector3 dir, Vector3 up)
        {
            if (idx >= 0 && idx < debugAnimationDirections.Length)
            {
                debugAnimationDirections[idx] = dir;
                debugUpDirections[idx] = up;
            }
        }

        private void CopyRotationIfPresent(int targetIdx, int sourceIdx)
        {
            if (IsAliasedFinger(targetIdx) || IsAliasedFinger(sourceIdx)) return;
            boneTransforms[targetIdx].rotation = boneTransforms[sourceIdx].rotation;
        }

        private void CacheBodyRestPose()
        {
            restLocalRotations = new Quaternion[boneTransforms.Length];
            for (int i = 0; i < boneTransforms.Length; i++)
            {
                if (boneTransforms[i] != null)
                    restLocalRotations[i] = boneTransforms[i].localRotation;
            }
        }

        public void ApplyOfflineBodyData(float[] limbData, long timestamp)
        {
            livePoseAnchored = false;
            ProcessBodyPose(limbData, timestamp, false);
        }

        // Used only by the live receiver. Recorded files retain their original mapping.
        public void ApplyBodyData(float[] limbData, long timestamp, bool fullBody = false, bool upperBody = false)
        {
            ProcessBodyPose(limbData, timestamp, true);
        }

        public void ResetLiveBodyTracking()
        {
            livePoseAnchored = false;
            lastLiveTimestamp = 0;
        }

        private void ProcessBodyPose(float[] limbData, long timestamp, bool live)
        {
            if (limbData == null || limbData.Length != 99)
            {
                Debug.LogError($"Data length mismatch: Expected 99, got {limbData?.Length ?? 0}");
                return;
            }
            foreach (float value in limbData)
                if (float.IsNaN(value) || float.IsInfinity(value)) return;
            lastRawBodyData = limbData;
            lastBodyTimestamp = timestamp;
            if (!bodyInitialized || boneTransforms == null || boneTransforms[0] == null)
            {
                if (!bodyInitWarningLogged)
                {
                    Debug.LogError("[RealTimeLhzAvatarEndpoint] 未初始化：avatarRoot 未赋值或关键骨骼缺失，已跳过姿势更新。请检查 Inspector 中的 avatarRoot 引用。");
                    bodyInitWarningLogged = true;
                }
                return;
            }
            if (live)
            {
                // Long tracking loss or sender restart establishes a new camera
                // origin at the current scene position, avoiding a reacquisition jump.
                if (Time.unscaledTime - lastLiveFrameTime > 1f ||
                    (timestamp > 0 && lastLiveTimestamp > 0 &&
                     (timestamp <= lastLiveTimestamp || timestamp - lastLiveTimestamp > 1000)))
                    livePoseAnchored = false;
                lastLiveFrameTime = Time.unscaledTime;
                lastLiveTimestamp = timestamp;
            }
            UpdateBodyModel(limbData, live);
        }

        private void InitializeBodyCalibration()
        {
            bodyForward = SafeTriangleNormal(boneTransforms[0].position, boneTransforms[18].position, boneTransforms[21].position);
            if (bodyForward.sqrMagnitude < 0.0001f) bodyForward = Vector3.forward;
            bodyForward.Normalize();
            restBodyForward = bodyForward; // Save for animation consistency check

            // If finger fallbacks produced duplicate bones, offset them slightly
            // so hand-normal and direction vectors are non-zero
            EnsureDistinctFingerBones();

            hipsCalibration = Quaternion.Inverse(boneTransforms[0].rotation) * SafeLookRotation(bodyForward);
            leftUpperLegCalibration = Quaternion.Inverse(boneTransforms[18].rotation) * SafeLookRotation((boneTransforms[18].position - boneTransforms[19].position), bodyForward);
            rightUpperLegCalibration = Quaternion.Inverse(boneTransforms[21].rotation) * SafeLookRotation((boneTransforms[21].position - boneTransforms[22].position), bodyForward);
            leftLowerLegCalibration = Quaternion.Inverse(boneTransforms[19].rotation) * SafeLookRotation((boneTransforms[19].position - boneTransforms[20].position), bodyForward);
            rightLowerLegCalibration = Quaternion.Inverse(boneTransforms[22].rotation) * SafeLookRotation((boneTransforms[22].position - boneTransforms[23].position), bodyForward);
            leftFootCalibration = Quaternion.Inverse(boneTransforms[20].rotation) * SafeLookRotation((boneTransforms[20].position - boneTransforms[25].position), bodyForward);
            rightFootCalibration = Quaternion.Inverse(boneTransforms[23].rotation) * SafeLookRotation((boneTransforms[23].position - boneTransforms[26].position), bodyForward);

            spineCalibration = Quaternion.Inverse(boneTransforms[1].rotation) * SafeLookRotation((boneTransforms[1].position - boneTransforms[2].position), bodyForward);
            chestCalibration = Quaternion.Inverse(boneTransforms[2].rotation) * SafeLookRotation((boneTransforms[2].position - boneTransforms[3].position), bodyForward);
            upperChestCalibration = Quaternion.Inverse(boneTransforms[3].rotation) * SafeLookRotation((boneTransforms[3].position - boneTransforms[24].position), bodyForward);
            leftShoulderCalibration = Quaternion.Inverse(boneTransforms[4].rotation) * SafeLookRotation((boneTransforms[4].position - boneTransforms[5].position), bodyForward);
            leftUpperArmCalibration = Quaternion.Inverse(boneTransforms[5].rotation) * SafeLookRotation((boneTransforms[5].position - boneTransforms[6].position), bodyForward);
            leftLowerArmCalibration = Quaternion.Inverse(boneTransforms[6].rotation) * SafeLookRotation((boneTransforms[6].position - boneTransforms[7].position), bodyForward);

            leftHandNormal = SafeTriangleNormal(boneRestPositions[7], boneRestPositions[9], boneRestPositions[8]);
            leftHandCalibration = Quaternion.Inverse(boneTransforms[7].rotation) * SafeLookRotation(boneRestPositions[8] - boneRestPositions[9], leftHandNormal);

            rightShoulderCalibration = Quaternion.Inverse(boneTransforms[11].rotation) * SafeLookRotation((boneTransforms[11].position - boneTransforms[12].position), bodyForward);
            rightUpperArmCalibration = Quaternion.Inverse(boneTransforms[12].rotation) * SafeLookRotation((boneTransforms[12].position - boneTransforms[13].position), bodyForward);
            rightLowerArmCalibration = Quaternion.Inverse(boneTransforms[13].rotation) * SafeLookRotation((boneTransforms[13].position - boneTransforms[14].position), bodyForward);

            rightHandNormal = SafeTriangleNormal(boneRestPositions[14], boneRestPositions[15], boneRestPositions[16]);
            rightHandCalibration = Quaternion.Inverse(boneTransforms[14].rotation) * SafeLookRotation(boneRestPositions[15] - boneRestPositions[16], rightHandNormal);

            leftThumbCalibration = Quaternion.Inverse(boneTransforms[8].rotation) * SafeLookRotation(boneRestPositions[8] - boneRestPositions[27], leftHandNormal);
            leftIndexCalibration = Quaternion.Inverse(boneTransforms[9].rotation) * SafeLookRotation(boneRestPositions[9] - boneRestPositions[29], leftHandNormal);
            leftLittleCalibration = Quaternion.Inverse(boneTransforms[10].rotation) * SafeLookRotation(boneRestPositions[10] - boneRestPositions[31], leftHandNormal);
            rightThumbCalibration = Quaternion.Inverse(boneTransforms[15].rotation) * SafeLookRotation(boneRestPositions[15] - boneRestPositions[28], rightHandNormal);
            rightIndexCalibration = Quaternion.Inverse(boneTransforms[16].rotation) * SafeLookRotation(boneRestPositions[16] - boneRestPositions[30], rightHandNormal);
            rightLittleCalibration = Quaternion.Inverse(boneTransforms[17].rotation) * SafeLookRotation(boneRestPositions[17] - boneRestPositions[32], rightHandNormal);

            headGaze = bodyForward;
            neckCalibration = Quaternion.Inverse(boneTransforms[24].rotation) * SafeLookRotation(headGaze);

            // Mirror the named Mids into the per-bone debug array (indexed by boneTransforms) so the
            // capture tool can read what rest calibration each joint used.
            debugRestRotations[0] = hipsCalibration;
            debugRestRotations[1] = spineCalibration; debugRestRotations[2] = chestCalibration; debugRestRotations[3] = upperChestCalibration;
            debugRestRotations[4] = leftShoulderCalibration; debugRestRotations[5] = leftUpperArmCalibration; debugRestRotations[6] = leftLowerArmCalibration; debugRestRotations[7] = leftHandCalibration;
            debugRestRotations[8] = leftThumbCalibration; debugRestRotations[9] = leftIndexCalibration; debugRestRotations[10] = leftLittleCalibration;
            debugRestRotations[11] = rightShoulderCalibration; debugRestRotations[12] = rightUpperArmCalibration; debugRestRotations[13] = rightLowerArmCalibration; debugRestRotations[14] = rightHandCalibration;
            debugRestRotations[15] = rightThumbCalibration; debugRestRotations[16] = rightIndexCalibration; debugRestRotations[17] = rightLittleCalibration;
            debugRestRotations[18] = leftUpperLegCalibration; debugRestRotations[19] = leftLowerLegCalibration; debugRestRotations[20] = leftFootCalibration;
            debugRestRotations[21] = rightUpperLegCalibration; debugRestRotations[22] = rightLowerLegCalibration; debugRestRotations[23] = rightFootCalibration;
            debugRestRotations[24] = neckCalibration;
        }

        private void EnsureLiveGrounding()
        {
            if (liveGrounding == null)
            {
                liveGrounding = GetComponent<LiveAvatarGrounding>();
                if (liveGrounding == null)
                    liveGrounding = gameObject.AddComponent<LiveAvatarGrounding>();
            }
            if (!liveGrounding.isActiveAndEnabled)
                liveGrounding.enabled = true;
            liveGrounding.detectJumpFromHeadHeight = true;
            liveGrounding.Initialize(boneTransforms[0], avatarRoot);
        }

        private void UpdateBodyModel(float[] data, bool live = false)
        {
            float[] scaledData = new float[data.Length];
            for (int i = 0; i < data.Length; i += 3)
            {
                if (i + 2 < data.Length)
                {
                    scaledData[i] = data[i] / 100f + poseOffsetX;
                    scaledData[i + 1] = data[i + 1] / 100f + poseOffsetY;
                    scaledData[i + 2] = data[i + 2] / 300f + poseOffsetZ;
                }
            }
            if (live)
            {
                if (enableGrounding) EnsureLiveGrounding();

                Vector3 inputHips = new Vector3((scaledData[69] + scaledData[72]) * 0.5f,
                    (scaledData[70] + scaledData[73]) * 0.5f, (scaledData[71] + scaledData[74]) * 0.5f);
                if (!livePoseAnchored)
                {
                    liveInputOrigin = inputHips;
                    liveSceneOrigin = boneTransforms[0].position;
                    lastLiveBodyForward = bodyForward.sqrMagnitude > 0.0001f ? bodyForward : restBodyForward;
                    livePoseAnchored = true;
                    if (liveGrounding != null) liveGrounding.ResetObservation();
                    Debug.Log("[RealTimeLhzAvatarEndpoint] Live movement anchored at current scene hips.");
                }
                Vector3 desired = liveSceneOrigin + inputHips - liveInputOrigin;
                if (enableGrounding && liveGrounding != null)
                {
                    // Horizontal motion follows the camera, but the vertical
                    // position is owned by the CharacterController: gravity,
                    // real ground collision and jump landing.
                    Vector3 groundedHips = liveGrounding.Observe(desired, scaledData);
                    TranslateLivePose(scaledData, inputHips, groundedHips);
                }
                else
                {
                    // Rebase every point equally so limb directions remain unchanged.
                    TranslateLivePose(scaledData, inputHips, desired);
                }
            }
            lastScaledBodyData = scaledData;

            boneTransforms[0].position = new Vector3((scaledData[69] + scaledData[72]) / 2.0f, (scaledData[70] + scaledData[73]) / 2.0f, (scaledData[71] + scaledData[74]) / 2.0f);
            leftUpperLegPosition = new Vector3(4.0f / 5.0f * scaledData[69] + 1.0f / 5.0f * scaledData[75], 4.0f / 5.0f * scaledData[70] + 1.0f / 5.0f * scaledData[76], 4.0f / 5.0f * scaledData[71] + 1.0f / 5.0f * scaledData[77]);
            rightUpperLegPosition = new Vector3(4.0f / 5.0f * scaledData[72] + 1.0f / 5.0f * scaledData[78], 4.0f / 5.0f * scaledData[73] + 1.0f / 5.0f * scaledData[79], 4.0f / 5.0f * scaledData[74] + 1.0f / 5.0f * scaledData[80]);

            if (live)
            {
                if (TryLivePelvisForward(boneTransforms[0].position, leftUpperLegPosition, rightUpperLegPosition,
                    out Vector3 measuredForward))
                {
                    if (Vector3.Dot(measuredForward, restBodyForward) < 0f) measuredForward = -measuredForward;
                    lastLiveBodyForward = measuredForward;
                }
                bodyForward = lastLiveBodyForward;
            }
            else bodyForward = SafeTriangleNormal(boneTransforms[0].position, leftUpperLegPosition, rightUpperLegPosition);
            if (bodyForward.sqrMagnitude < 0.0001f) bodyForward = restBodyForward;
            else
            {
                bodyForward.Normalize();
                // CRITICAL: ensure animation bodyForward matches rest bodyForward hemisphere
                // If they point in opposite directions, all bones flip 180°
                if (Vector3.Dot(bodyForward, restBodyForward) < 0)
                    bodyForward = -bodyForward;
            }
            SetBone(0, bodyForward, hipsCalibration);

            // REMOVED the 180 degree hacks from the legs!
            leftUpperLegDirection = leftUpperLegPosition - new Vector3(scaledData[75], scaledData[76], scaledData[77]);
            SetBone(18, leftUpperLegDirection, bodyForward, leftUpperLegCalibration);

            rightUpperLegDirection = rightUpperLegPosition - new Vector3(scaledData[78], scaledData[79], scaledData[80]);
            SetBone(21, rightUpperLegDirection, bodyForward, rightUpperLegCalibration);

            leftLowerLegDirection = new Vector3(scaledData[75], scaledData[76], scaledData[77]) - new Vector3(scaledData[81], scaledData[82], scaledData[83]);
            SetBone(19, leftLowerLegDirection, bodyForward, leftLowerLegCalibration);

            rightLowerLegDirection = new Vector3(scaledData[78], scaledData[79], scaledData[80]) - new Vector3(scaledData[84], scaledData[85], scaledData[86]);
            SetBone(22, rightLowerLegDirection, bodyForward, rightLowerLegCalibration);

            leftFootDirection = new Vector3(scaledData[81], scaledData[82], scaledData[83]) - new Vector3(scaledData[93], scaledData[94], scaledData[95]);
            SetBone(20, leftFootDirection, bodyForward, leftFootCalibration);

            rightFootDirection = new Vector3(scaledData[84], scaledData[85], scaledData[86]) - new Vector3(scaledData[96], scaledData[97], scaledData[98]);
            SetBone(23, rightFootDirection, bodyForward, rightFootCalibration);

            // The rest of the body remains unchanged
            spineDirection = boneTransforms[0].position - new Vector3((scaledData[33] + scaledData[36]) / 2.0f, (scaledData[34] + scaledData[37]) / 2.0f, (scaledData[35] + scaledData[38]) / 2.0f);
            SetBone(1, spineDirection, bodyForward, spineCalibration);
            SetBone(2, spineDirection, bodyForward, chestCalibration);
            SetBone(3, spineDirection, bodyForward, upperChestCalibration);

            leftShoulderDirection = new Vector3(scaledData[36] - scaledData[33], scaledData[37] - scaledData[34], scaledData[38] - scaledData[35]);
            SetBone(4, leftShoulderDirection, bodyForward, leftShoulderCalibration);

            rightShoulderDirection = new Vector3(scaledData[33] - scaledData[36], scaledData[34] - scaledData[37], scaledData[35] - scaledData[38]);
            SetBone(11, rightShoulderDirection, bodyForward, rightShoulderCalibration);

            leftUpperArmDirection = new Vector3(scaledData[33] - scaledData[39], scaledData[34] - scaledData[40], scaledData[35] - scaledData[41]);
            SetBone(5, leftUpperArmDirection, bodyForward, leftUpperArmCalibration);

            rightUpperArmDirection = new Vector3(scaledData[36] - scaledData[42], scaledData[37] - scaledData[43], scaledData[38] - scaledData[44]);
            SetBone(12, rightUpperArmDirection, bodyForward, rightUpperArmCalibration);

            leftLowerArmDirection = new Vector3(scaledData[39] - scaledData[45], scaledData[40] - scaledData[46], scaledData[41] - scaledData[47]);
            SetBone(6, leftLowerArmDirection, bodyForward, leftLowerArmCalibration);

            rightLowerArmDirection = new Vector3(scaledData[42] - scaledData[48], scaledData[43] - scaledData[49], scaledData[44] - scaledData[50]);
            SetBone(13, rightLowerArmDirection, bodyForward, rightLowerArmCalibration);

            leftIndexPosition = new Vector3(scaledData[57], scaledData[58], scaledData[59]);
            leftThumbPosition = new Vector3(scaledData[63], scaledData[64], scaledData[65]);
            leftLittlePosition = new Vector3(scaledData[51], scaledData[52], scaledData[53]);
            leftHandNormal = SafeTriangleNormal(new Vector3(scaledData[45], scaledData[46], scaledData[47]), leftIndexPosition, leftThumbPosition);
            SetBone(7, leftThumbPosition - leftIndexPosition, leftHandNormal, leftHandCalibration);

            rightIndexPosition = new Vector3(scaledData[60], scaledData[61], scaledData[62]);
            rightThumbPosition = new Vector3(scaledData[66], scaledData[67], scaledData[68]);
            rightLittlePosition = new Vector3(scaledData[54], scaledData[55], scaledData[56]);
            rightHandNormal = SafeTriangleNormal(new Vector3(scaledData[48], scaledData[49], scaledData[50]), rightThumbPosition, rightIndexPosition);
            SetBone(14, rightThumbPosition - rightIndexPosition, rightHandNormal, rightHandCalibration);

            leftThumbDirection = new Vector3(scaledData[45], scaledData[46], scaledData[47]) - leftThumbPosition;
            SetBone(8, leftThumbDirection, leftHandNormal, leftThumbCalibration);
            leftIndexDirection = new Vector3(scaledData[45], scaledData[46], scaledData[47]) - leftIndexPosition;
            SetBone(9, leftIndexDirection, leftHandNormal, leftIndexCalibration);
            leftLittleDirection = new Vector3(scaledData[45], scaledData[46], scaledData[47]) - leftLittlePosition;
            SetBone(10, leftLittleDirection, leftHandNormal, leftLittleCalibration);

            CopyRotationIfPresent(33, 9);
            CopyRotationIfPresent(35, 9);

            rightThumbDirection = new Vector3(scaledData[48], scaledData[49], scaledData[50]) - rightThumbPosition;
            SetBone(15, rightThumbDirection, rightHandNormal, rightThumbCalibration);
            rightIndexDirection = new Vector3(scaledData[48], scaledData[49], scaledData[50]) - rightIndexPosition;
            SetBone(16, rightIndexDirection, rightHandNormal, rightIndexCalibration);
            rightLittleDirection = new Vector3(scaledData[48], scaledData[49], scaledData[50]) - rightLittlePosition;
            SetBone(17, rightLittleDirection, rightHandNormal, rightLittleCalibration);

            CopyRotationIfPresent(34, 16);
            CopyRotationIfPresent(36, 16);

            headPosition = new Vector3((scaledData[21] + scaledData[24]) / 2.0f, (scaledData[22] + scaledData[25]) / 2.0f, (scaledData[23] + scaledData[26]) / 2.0f);
            headGaze = new Vector3(scaledData[0], scaledData[1], scaledData[2]) - headPosition;
            SetBone(24, headGaze, neckCalibration);

            hasBodyPose = true;
        }

        private static void TranslateLivePose(float[] pose, Vector3 inputOrigin, Vector3 sceneOrigin)
        {
            Vector3 shift = sceneOrigin - inputOrigin;
            // Translate every point equally: the solver mixes hips with shoulders
            // and knees, so rebasing hips alone would change limb directions.
            for (int i = 0; i < pose.Length; i += 3)
            {
                pose[i] += shift.x;
                pose[i + 1] += shift.y;
                pose[i + 2] += shift.z;
            }
        }

        private static bool TryLivePelvisForward(Vector3 hips, Vector3 left, Vector3 right, out Vector3 result)
        {
            Vector3 a = hips - left, b = hips - right;
            Vector3 cross = Vector3.Cross(a, b);
            float product = a.sqrMagnitude * b.sqrMagnitude;
            // Relative degeneracy check, independent of camera distance / body size.
            // Invalid triangles retain the last reliable direction, never Vector3.up.
            result = Vector3.zero;
            if (product < 1e-12f || cross.sqrMagnitude <= product * 0.0001f) return false;
            result = cross.normalized;
            return true;
        }

        private Vector3 TriangleNormal(Vector3 a, Vector3 b, Vector3 c)
        {
            return SafeTriangleNormal(a, b, c);
        }

        private void SetupBodyAxisCorrection()
        {
            if (enableAxisCorrection)
            {
                bodyAxisCorrectionRoot = new GameObject($"{avatarRoot.name}_RealTimeLhzAxisCorrection");
                originalAvatarParent = avatarRoot.transform.parent;
                originalAvatarLocalPosition = avatarRoot.transform.localPosition;
                originalAvatarLocalRotation = avatarRoot.transform.localRotation;
                Vector3 originalPosition = avatarRoot.transform.position;
                Quaternion originalRotation = avatarRoot.transform.rotation;
                bodyAxisCorrectionRoot.transform.position = Vector3.zero;
                bodyAxisCorrectionRoot.transform.rotation = Quaternion.identity;
                avatarRoot.transform.SetParent(bodyAxisCorrectionRoot.transform, false);
                avatarRoot.transform.position = originalPosition;
                avatarRoot.transform.rotation = originalRotation;
                bodyAxisCorrectionRoot.AddComponent<RealTimeLhzAxisCorrection>();
            }
        }

        private Transform FindBodyBoneRecursive(Transform parent, string name)
        {
            if (parent.name == name) return parent;
            foreach (Transform child in parent)
            {
                Transform found = FindBodyBoneRecursive(child, name);
                if (found != null) return found;
            }
            return null;
        }

        private void DestroyAxisCorrectionRoot()
        {
            if (bodyAxisCorrectionRoot == null) return;
            if (avatarRoot != null && avatarRoot.transform.parent == bodyAxisCorrectionRoot.transform)
            {
                avatarRoot.transform.SetParent(originalAvatarParent, false);
                avatarRoot.transform.localPosition = originalAvatarLocalPosition;
                avatarRoot.transform.localRotation = originalAvatarLocalRotation;
            }
            Destroy(bodyAxisCorrectionRoot);
            bodyAxisCorrectionRoot = null;
            originalAvatarParent = null;
        }
    }

    public sealed class RealTimeLhzAxisCorrection : MonoBehaviour
    {
        private void LateUpdate()
        {
            transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        }
    }
}
