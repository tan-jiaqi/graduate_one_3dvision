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
        // �ڱ༭����ֹͣ����ģʽ
        UnityEditor.EditorApplication.isPlaying = false;
        #else
        // �ڹ�������Ϸ���˳�
        Application.Quit();
        #endif
    }
}
