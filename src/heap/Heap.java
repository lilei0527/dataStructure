package heap;


/**
 * @author lilei
 * 堆
 * 时间复杂度O(n)
 * 空间复杂度O(1)
 **/
public class Heap {

    private void buildHeap(int[] arr, boolean isMaxHeap) {
        for (int i = arr.length / 2; i >= 0; i--) {
            heap(arr, i, isMaxHeap);
        }
    }

    private void heap(int[] a, int i, boolean isMaxHeap) {
        int len = a.length;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int bound = i;

        if (left < len && a[left] > a[bound] && isMaxHeap) {
            bound = left;
        }

        if (right < len && a[right] > a[bound] && isMaxHeap) {
            bound = right;
        }

        if (left < len && a[left] < a[bound] && !isMaxHeap) {
            bound = left;
        }

        if (right < len && a[right] < a[bound] && !isMaxHeap) {
            bound = right;
        }

        if (bound != i) {
            swap(i, bound, a);
            heap(a, bound, isMaxHeap);
        }
    }


    void swap(int left, int right, int[] a) {
        int temp;
        temp = a[left];
        a[left] = a[right];
        a[right] = temp;
    }
}
