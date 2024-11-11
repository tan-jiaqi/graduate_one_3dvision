using UnityEngine;

public class Quitgame : MonoBehaviour
{
    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Quit();
        }
    }
    public void Quit()
    {
        #if UNITY_EDITOR
        // 在编辑器中停止播放模式
        UnityEditor.EditorApplication.isPlaying = false;
        #else
        // 在构建的游戏中退出
        Application.Quit();
        #endif
    }
}
