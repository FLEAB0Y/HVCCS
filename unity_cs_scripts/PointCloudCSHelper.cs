using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PointCloudCSHelper : MonoBehaviour
{
    public Material material;

    ComputeBuffer pointPosBuffer;
    ComputeBuffer pointColBuffer;
    
    int pointCount; // 实际点的数量
    
    int PointCloudPotNum { get { return PointCloud.Width * PointCloud.Height; } }

    void Start()
    {
        // 检查材质是否正确设置
        if (material == null)
        {
            Debug.LogError("点云材质未设置！");
            
            // 尝试从当前对象获取材质
            MeshRenderer renderer = GetComponent<MeshRenderer>();
            if (renderer != null && renderer.material != null)
            {
                material = renderer.material;
                Debug.Log("已自动设置材质");
            }
            else
            {
                return;
            }
        }
        
        // 初始创建一个空点以避免错误
        Vector3[] initialPositions = new Vector3[1] { Vector3.zero };
        pointCount = 1;
        
        // 创建初始缓冲区
        pointPosBuffer = new ComputeBuffer(pointCount, 12);
        pointColBuffer = new ComputeBuffer(pointCount, 16);
        
        Color[] initialColors = new Color[1] { Color.white };

        // 设置初始数据到缓冲区
        pointPosBuffer.SetData(initialPositions);
        pointColBuffer.SetData(initialColors);

        // 设置材质缓冲区
        material.SetBuffer("PointPos", pointPosBuffer);
        material.SetBuffer("PointCol", pointColBuffer);

        // 更新MeshFilter中的点数量
        UpdateMeshPointCount();
    }

    // 更新MeshFilter中的点数量
    private void UpdateMeshPointCount()
    {
        PointCloud pointCloud = GetComponent<PointCloud>();
        if (pointCloud != null)
        {
            pointCloud.UpdatePointCount(pointCount);
        }
    }

    void Update()
    {
    }

    void OnDestroy()
    {
        if (pointPosBuffer != null)
        {
            pointPosBuffer.Release();
            pointPosBuffer.Dispose();
        }

        if (pointColBuffer != null)
        {
            pointColBuffer.Release();
            pointColBuffer.Dispose();
        }
    }

    // 从Socket接收到的数据更新点云
    public void UpdatePointCloudData(Vector3[] positions, Color[] colors)
    {
        if (positions == null || colors == null || positions.Length != colors.Length)
        {
            Debug.LogError("无效的点云数据");
            return;
        }

        // 更新点数量
        int newPointCount = positions.Length;
        
        // 确保点数不超过最大限制
        if (newPointCount > PointCloudPotNum)
        {
            Debug.LogWarning($"点云数据点数({newPointCount})超过最大限制({PointCloudPotNum})，将截断多余点");
            newPointCount = PointCloudPotNum;
        }
        
        // 释放旧的缓冲区
        if (pointPosBuffer != null)
        {
            pointPosBuffer.Release();
            pointPosBuffer.Dispose();
        }
        
        if (pointColBuffer != null)
        {
            pointColBuffer.Release();
            pointColBuffer.Dispose();
        }
        
        // 创建新的缓冲区
        pointPosBuffer = new ComputeBuffer(newPointCount, 12);
        pointColBuffer = new ComputeBuffer(newPointCount, 16);
        
        // 更新点数量
        pointCount = newPointCount;
        
        // 转换坐标系（如果需要）
        Vector3[] worldPositions = new Vector3[positions.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            // 将点转换为本地坐标系，确保它们相对于对象自身而不是相机
            worldPositions[i] = positions[i];
        }
        
        // 直接设置处理后的数据到缓冲区
        pointPosBuffer.SetData(worldPositions);
        pointColBuffer.SetData(colors, 0, 0, newPointCount);
        
        // 重新设置材质缓冲区
        material.SetBuffer("PointPos", pointPosBuffer);
        material.SetBuffer("PointCol", pointColBuffer);
        
        // 更新MeshFilter中的点数量
        UpdateMeshPointCount();
        
        Debug.Log($"点云数据已更新，点数: {newPointCount}");
    }
}
