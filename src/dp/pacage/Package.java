package dp.pacage;

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
        if (weight.length != value.length) {
            throw new RuntimeException("weight length must equal value length");
        }
    }

    //cap:容量
    public int getMaxValue(int cap) {
        Strategy strategy = 1 << weight.length > cap * weight.length ? new DynamicStrategy() : new RecursionStrategy();
        return strategy.getMaxValue(cap, weight, value);
    }

    public static void main(String[] args) {
        int[] v = {400, 500, 200, 300, 350,100,200,900};
        int[] w = {5, 5, 3, 4, 3,1,2,7};


        Package p = new Package(w, v);
        int maxValue = p.getMaxValue(30);
        System.out.println(maxValue);


        long t1 = System.currentTimeMillis();
        DynamicStrategy dynamicStrategy = new DynamicStrategy();
        System.out.println(dynamicStrategy.getMaxValue(30, w, v));
        long t2 = System.currentTimeMillis();


        long t3 = System.currentTimeMillis();
        RecursionStrategy recursionStrategy = new RecursionStrategy();
        System.out.println(recursionStrategy.getMaxValue(30, w, v));
        long t4 = System.currentTimeMillis();


        System.out.println((t2 - t1));
        System.out.println((t4 - t3));
    }
}