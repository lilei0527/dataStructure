package dp;

/**
 * @author lilei
 * 01背包问题
 **/
public class Package {
    private int[] weight;
    private int[] value;

    public Package(int[] weight, int[] value) {
        this.weight = weight;
        this.value = value;
    }

    //cap:容量
    public int getMaxValue(int cap) {
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

    public static void main(String[] args) {
        int[] v = {400, 500, 200, 300, 350};
        int[] w = {5, 5, 3, 4, 3};
        Package p = new Package(w, v);
        int maxValue = p.getMaxValue(9);
        System.out.println(maxValue);
    }
}