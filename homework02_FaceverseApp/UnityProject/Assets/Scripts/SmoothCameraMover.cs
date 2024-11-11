using UnityEngine;

public class SmoothCameraMover : MonoBehaviour
{
    public Transform pointA; // ��ʼ��
    public Transform pointB; // �յ�

    private float duration = 0.6f; // �ƶ�����ʱ��
    private bool isMoving = false;
    private int target = 0;

    private void Start()
    {
        transform.position = pointA.transform.position;
    }

    void Update()
    {
        // ��ⰴť����������滻Ϊ��İ�ť����߼���
        if (Input.GetKeyDown(KeyCode.Space) && !isMoving && target==0)
        {
            StartCoroutine(MoveCamera(pointA.position, pointB.position, duration));
        }else if(Input.GetKeyDown(KeyCode.Space) && !isMoving && target == 1)
        {
            StartCoroutine(MoveCamera(pointB.position, pointA.position, duration));
        }
    }

    private System.Collections.IEnumerator MoveCamera(Vector3 start, Vector3 end, float duration)
    {
        isMoving = true;
        float elapsedTime = 0f;

        while (elapsedTime < duration)
        {
            float t = elapsedTime / duration;
            t = Mathf.SmoothStep(0, 1, t); // ʹ��SmoothStep���л��뻺����ֵ

            transform.position = Vector3.Lerp(start, end, t);
            elapsedTime += Time.deltaTime;
            yield return null;
        }

        transform.position = end;
        isMoving = false;
        target += 1;
        target %= 2;
    }
}
