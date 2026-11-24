# RealTimeLhzAvatarEndpoint

`RealTimeLhzAvatarEndpoint.cs` is the complete Unity-side endpoint for one
digital human. Attach the same component to each avatar instance. Configure
only the avatar references and ports; do not modify the code for different
avatars.

## Unity configuration

Assign:

- `Avatar Root`: the digital human root containing the `mixamorig:*` bones.
- `Face Renderer`: the `SkinnedMeshRenderer` containing the 52 BlendShapes.
- `Listen Port`: the Socket port emitted by `grpc2socket.py`.
- `Feedback Port`: the Socket port plus the feedback offset (default `+1000`).

The default mappings in `grpc2socket.py` are:

| gRPC input | Unity data port | Unity feedback port |
|---:|---:|---:|
| 50051 | 8890 | 9890 |
| 50052 | 8891 | 9891 |
| 50053 | 8892 | 9892 |
| 50054 | 8893 | 9893 |

Each endpoint must use a unique data port and the matching feedback port. A
single input stream is intentionally delivered to one endpoint only.

## Wire format

The component expects one UTF-8 line per frame from the persistent TCP
connection:

```text
timestamp,52 face values,99 pose values\n
```

It sends one UTF-8 feedback message per applied frame:

```text
timing:t_begin,t_transit,t_final
```

The body driver uses the lhz Mixamo names (`mixamorig:Hips`,
`mixamorig:Spine1`, `mixamorig:LeftArm`, and so on), while the receiver and
feedback timing follow the real-time Unity protocol.

The endpoint keeps bounded receive and feedback queues (240 entries by
default). If Unity cannot process the incoming 120 FPS stream, it drops the
oldest queued frame instead of allowing latency and memory usage to grow
without bound. Set `Max Queued Frames` and `Max Pending Feedbacks` in the
Inspector for a different overload limit.
