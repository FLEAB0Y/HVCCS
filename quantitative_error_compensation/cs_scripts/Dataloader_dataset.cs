using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace DatasetScripts
{
    public class DataLoader : MonoBehaviour
    {
        // 文件配置
        [SerializeField] private string dataDirectory = "proc_features"; // 相对于Assets目录的路径或绝对路径
        [SerializeField] private float frameRate = 30f; // 每秒播放多少帧
        
        // 引用BlendShape控制器
        [SerializeField] private BSCtrl_dataset blendShapeController;
        
        // 添加BDCtrl控制器引用
        [SerializeField] private BDCtrl_dataset neZhaMov;

        // 数据帧存储
        private List<string> dataFrames = new List<string>();
        private int currentFrameIndex = 0;
        private float frameTimer = 0f;
        private bool isPlaying = false;
        
        // 文件控制
        private List<string> txtFiles = new List<string>();
        private int currentFileIndex = 0;
        private float fileTransitionTimer = 0f;
        private bool isWaitingForNextFile = false;
        [SerializeField] [Tooltip("文件之间的切换延迟时间（秒）")] 
        private float fileSwitchDelay = 10f; // 文件之间的间隔，单位为秒
        
        // 界面控制选项
        [SerializeField] private bool autoPlay = true;
        [SerializeField] private bool loopPlayback = true;

        void Start()
        {
            // 如果没有手动指定BlendShape控制器，则尝试查找
            if (blendShapeController == null)
            {
                blendShapeController = GetComponent<BSCtrl_dataset>();
                if (blendShapeController == null)
                {
                    blendShapeController = FindObjectOfType<BSCtrl_dataset>();
                    if (blendShapeController == null)
                    {
                        Debug.LogError("未找到BlendShape控制器，请手动指定");
                    }
                }
            }
            
            // 如果没有手动指定BDCtrl控制器，则尝试查找
            if (neZhaMov == null)
            {
                neZhaMov = GetComponent<BDCtrl_dataset>();
                if (neZhaMov == null)
                {
                    neZhaMov = FindObjectOfType<BDCtrl_dataset>();
                    if (neZhaMov == null)
                    {
                        Debug.LogError("未找到BDCtrl控制器，请手动指定");
                    }
                }
            }
            
            // 获取所有TXT文件并按名称排序
            FindAllTxtFiles();
            
            // 如果有文件，加载第一个
            if (txtFiles.Count > 0)
            {
                LoadDataFile(txtFiles[0]);
                
                // 如果设置了自动播放，则开始播放
                if (autoPlay && dataFrames.Count > 0)
                {
                    isPlaying = true;
                }
            }
            else
            {
                Debug.LogError($"【文件错误】在目录 {dataDirectory} 中没有找到任何.txt文件");
            }
        }

        // 查找所有TXT文件并按名称排序
        private void FindAllTxtFiles()
        {
            try
            {
                string fullPath;
                
                // 判断是相对路径还是绝对路径
                if (Path.IsPathRooted(dataDirectory))
                {
                    fullPath = dataDirectory;
                }
                else
                {
                    // 相对于项目根目录的路径
                    fullPath = Path.Combine(Application.dataPath, dataDirectory);
                }
                
                if (!Directory.Exists(fullPath))
                {
                    Debug.LogError($"【目录错误】找不到指定目录: {fullPath}");
                    return;
                }
                
                // 获取所有.txt文件并按名称排序
                txtFiles = Directory.GetFiles(fullPath, "*.txt")
                    .OrderBy(f => Path.GetFileName(f))
                    .ToList();
                
                // 移除不必要的日志输出
            }
            catch (Exception e)
            {
                Debug.LogError($"【文件错误】查找.txt文件失败: {e.Message}");
            }
        }

        // 加载数据文件
        private void LoadDataFile(string filePath)
        {
            try
            {
                // 清空之前的数据
                dataFrames.Clear();
                currentFrameIndex = 0;
                frameTimer = 0f;
                
                if (!File.Exists(filePath))
                {
                    Debug.LogError($"【文件错误】找不到数据文件: {filePath}");
                    return;
                }
                
                // 读取所有行
                string[] lines = File.ReadAllLines(filePath);
                
                // 过滤掉空行
                foreach (string line in lines)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        dataFrames.Add(line);
                    }
                }
                
                // 显示当前加载的文件名
                Debug.Log($"【文件加载】已加载 {Path.GetFileName(filePath)}");
            }
            catch (Exception e)
            {
                Debug.LogError($"【文件错误】加载数据文件失败: {e.Message}");
            }
        }
        
        // 在Update中处理帧播放和文件切换
        void Update()
        {
            // 检测空格键按下，切换播放/暂停状态
            if (Input.GetKeyDown(KeyCode.Space))
            {
                if (isPlaying)
                {
                    PauseAnimation();
                }
                else
                {
                    PlayAnimation();
                }
            }
            
            if (!isPlaying)
                return;
                
            // 如果正在等待切换到下一个文件
            if (isWaitingForNextFile)
            {
                fileTransitionTimer += Time.deltaTime;
                if (fileTransitionTimer >= fileSwitchDelay)
                {
                    // 重置计时器
                    fileTransitionTimer = 0f;
                    isWaitingForNextFile = false;
                    
                    // 加载下一个文件
                    currentFileIndex = (currentFileIndex + 1) % txtFiles.Count;
                    LoadDataFile(txtFiles[currentFileIndex]);
                    
                    // 显示文件切换信息
                    Debug.Log($"【文件切换】已切换到文件 {Path.GetFileName(txtFiles[currentFileIndex])}");
                }
                return;
            }
                
            if (dataFrames.Count == 0)
                return;
                
            // 计算帧间隔时间
            frameTimer += Time.deltaTime;
            float frameInterval = 1f / frameRate;
            
            // 如果达到了播放下一帧的时间
            if (frameTimer >= frameInterval)
            {
                // 处理当前帧
                ProcessFrame(dataFrames[currentFrameIndex]);
                
                // 更新帧索引
                currentFrameIndex++;
                
                // 如果到达末尾
                if (currentFrameIndex >= dataFrames.Count)
                {
                    // 如果还有其他文件要播放
                    if (txtFiles.Count > 1)
                    {
                        // 开始等待切换到下一个文件
                        isWaitingForNextFile = true;
                        fileTransitionTimer = 0f;
                    }
                    // 否则检查是否需要循环当前文件
                    else if (loopPlayback)
                    {
                        currentFrameIndex = 0;
                    }
                    else
                    {
                        isPlaying = false;
                    }
                }
                
                // 重置帧计时器（考虑超出的时间）
                frameTimer -= frameInterval;
            }
        }
        
        // 处理单帧数据
        private void ProcessFrame(string data)
        {
            try
            {
                // 解析逗号分隔的数据字符串
                string[] parts = data.Split(',');
                
                if (parts.Length < 151) // 至少需要52(面部数据) + 99(姿势数据) = 151
                {
                    Debug.LogWarning($"【数据不足】数据项不足，期望至少151项，实际为{parts.Length}项");
                    return;
                }
                
                // 提取面部表情数据（前52个数据项）
                float[] faceData = new float[52];
                for (int i = 0; i < 52 && i < parts.Length; i++)
                {
                    if (float.TryParse(parts[i], out float value))
                    {
                        faceData[i] = value;
                    }
                    else
                    {
                        Debug.LogWarning($"【解析错误】无法解析面部数据项 {i}: {parts[i]}");
                        faceData[i] = 0f;
                    }
                }
                
                // 提取姿势数据（后99个数据项）
                float[] limbData = new float[99];
                for (int i = 0; i < 99 && i + 52 < parts.Length; i++)
                {
                    if (float.TryParse(parts[i + 52], out float value))
                    {
                        limbData[i] = value;
                    }
                    else
                    {
                        Debug.LogWarning($"【解析错误】无法解析姿势数据项 {i}: {parts[i + 52]}");
                        limbData[i] = 0f;
                    }
                }
                
                // 将面部数据传递给BlendShape控制器
                if (blendShapeController != null)
                {
                    // 使用0作为时间戳，因为我们不再需要计算延迟
                    blendShapeController.ProcessBlendShapeDataArray(faceData, 0);
                }
                else
                {
                    Debug.LogError("【控制器缺失】BlendShape控制器未找到");
                }
                
                // 处理姿势数据
                if (neZhaMov != null)
                {
                    neZhaMov.ProcessLimbData(limbData, 0);
                }
                else
                {
                    Debug.LogError("【控制器缺失】BDCtrl控制器未找到");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"【解析错误】解析数据错误: {e.Message}\n{e.StackTrace}");
            }
        }
        
        // 公共控制方法
        public void PlayAnimation()
        {
            isPlaying = true;
        }
        
        public void PauseAnimation()
        {
            isPlaying = false;
        }
        
        public void StopAnimation()
        {
            isPlaying = false;
            currentFrameIndex = 0;
            frameTimer = 0f;
            isWaitingForNextFile = false;
            fileTransitionTimer = 0f;
        }
        
        public void SetFrame(int frameIndex)
        {
            if (frameIndex >= 0 && frameIndex < dataFrames.Count)
            {
                currentFrameIndex = frameIndex;
                frameTimer = 0f;
                if (!isPlaying)
                {
                    ProcessFrame(dataFrames[currentFrameIndex]);
                }
            }
        }
        
        public int GetTotalFrames()
        {
            return dataFrames.Count;
        }
        
        public int GetCurrentFrame()
        {
            return currentFrameIndex;
        }
        
        public bool IsPlaying()
        {
            return isPlaying;
        }
        
        public void ReloadCurrentFile()
        {
            if (txtFiles.Count > 0 && currentFileIndex < txtFiles.Count)
            {
                LoadDataFile(txtFiles[currentFileIndex]);
            }
        }
        
        public void ReloadAllFiles()
        {
            FindAllTxtFiles();
            if (txtFiles.Count > 0)
            {
                currentFileIndex = 0;
                LoadDataFile(txtFiles[0]);
            }
        }
        
        public string GetCurrentFileName()
        {
            if (txtFiles.Count > 0 && currentFileIndex < txtFiles.Count)
            {
                return Path.GetFileName(txtFiles[currentFileIndex]);
            }
            return string.Empty;
        }
        
        public int GetTotalFiles()
        {
            return txtFiles.Count;
        }
    }
}