using UnityEngine;
using System.IO;

[System.Serializable]
public class ShapeKey
{
    public int id;
    public float weight;
}

[System.Serializable]
public class ShapeKeyData
{
    public ShapeKey[] shapeKeys;
}

public class BlendShapeLoader : MonoBehaviour
{
    public SkinnedMeshRenderer skinnedMeshRenderer; // 需要在编辑器中分配
    public string filepath;

    void Start()
    {
        LoadShapeKeysFromJson(filepath);
    }

    void LoadShapeKeysFromJson(string filePath)
    {
        if (!File.Exists(filePath))
        {
            Debug.LogError("File not found: " + filePath);
            return;
        }

        string jsonContent = File.ReadAllText(filePath);
        ShapeKeyData shapeKeyData = JsonUtility.FromJson<ShapeKeyData>(jsonContent);

        foreach (var shapeKey in shapeKeyData.shapeKeys)
        {
            // 确保权重在0到100之间
            float clampedWeight = Mathf.Clamp(shapeKey.weight * 100, -100, 100);
            skinnedMeshRenderer.SetBlendShapeWeight(shapeKey.id -1, clampedWeight);
        }
    }
}
