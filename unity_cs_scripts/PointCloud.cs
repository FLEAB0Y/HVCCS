using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PointCloud : MonoBehaviour, IDragHandler
{
    public const int Width = 640, Height = 480;

    private Vector3[] positions;
    private int[] indices;
    private Color[] colors;
    private Mesh mesh;
    private int activePointCount;
    private Camera mainCamera;

    // Use this for initialization
    void Start()
    {
        var totalPointNum = Width * Height;
        positions = new Vector3[totalPointNum];
        indices = new int[totalPointNum];
        colors = new Color[totalPointNum];
        
        for (int i = 0; i < totalPointNum; i++)
        {
            positions[i] = new Vector3(0, 0, 0);
            indices[i] = i;
            colors[i] = Color.red;
        }
        
        mesh = GetComponent<MeshFilter>().mesh;
        mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.vertices = positions;
        mesh.colors = colors;
        mesh.SetIndices(indices, MeshTopology.Points, 0);
        mesh.MarkDynamic();
        
        // 默认使用所有点
        activePointCount = totalPointNum;
    }

    void Awake()
    {
        mainCamera = Camera.main;
    }

    void LateUpdate()
    {
        // 确保点云在Game视图中可见
        if (mainCamera != null)
        {
            // 可选：让点云始终面向相机
            // transform.LookAt(mainCamera.transform);
        }
    }

    // 更新实际使用的点数量
    public void UpdatePointCount(int pointCount)
    {
        if (pointCount <= 0 || pointCount > Width * Height)
        {
            Debug.LogError($"点数量 {pointCount} 超出有效范围 [1, {Width * Height}]");
            return;
        }

        this.activePointCount = pointCount;
        
        // 重新设置索引以仅使用活动点
        int[] newIndices = new int[pointCount];
        for (int i = 0; i < pointCount; i++)
        {
            newIndices[i] = i;
        }
        
        mesh.SetIndices(newIndices, MeshTopology.Points, 0);
        Debug.Log($"更新点云显示数量为: {pointCount}");
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (eventData.button == PointerEventData.InputButton.Right)
        {
            this.transform.Rotate(eventData.delta.y, eventData.delta.x, 0);
        }
    }
}
