package dp.pacage;

/**
 * @author lilei
 * 动态规划核心就在于拆分子问题，记住过往，减少重复计算
 * 动态规划解决01背包
 * 时间复杂度：cap*n (cap:背包容量,n:物品数量)
 **/
public class DynamicStrategy implements Strategy {
    @Override
    public int getMaxValue(int cap, int[] weight, int[] value) {
        int[] preResults = new int[cap + 1];
        int[] results = new int[cap + 1];

        //填充
        for (int i = 0; i < cap; i++) {
            if (i < weight[0]) {
                preResults[i] = 0;
            } else {
                preResults[i] = value[0];
            }
        }

        for (int i = 1; i < weight.length; i++) { //第几个物品
            for (int j = 0; j <= cap; j++) { //背包总重量
                if (j < weight[i]) {
                    results[j] = preResults[j];
                } else {
                    results[j] = Math.max(preResults[j], preResults[j - weight[i]] + value[i]);
                }
            }
            System.arraycopy(results, 0, preResults, 0, cap + 1);
        }
        return preResults[cap];
    }

    //一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 10 级的台阶总共有多少种跳法
    public int numWays(int n) {
        if (n<= 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int a = 1;
        int b = 2;
        int temp = 0;
        for (int i = 3; i <= n; i++) {
            temp = a + b;
            a = b;
            b = temp;
        }
        return temp;
    }
}
