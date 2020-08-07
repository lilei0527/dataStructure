package dp.array;

/**
 * @author lilei
 **/
public class Arrays {


    /**
     * @author lilei
     * create at 10:48 2020/7/28
     * 求给定数组连续子数组的最大值（dp求解）
     * 时间复杂度O(n) 空间复杂度O(1)
     **/
    public static int maxSubArray(int[] nums) {
        int res = nums[0];
        int max = 0;
        for (int i = 1; i < nums.length; i++) {
            max = Math.max(max + nums[i], 0);
            res = Math.max(max, res);
        }
        return res;
    }

    public static void main(String[] args) {
        int[] nums = {-1, 2, 1, -1, 2, -1,2};
        System.out.println(maxSubArray(nums));
    }
}
