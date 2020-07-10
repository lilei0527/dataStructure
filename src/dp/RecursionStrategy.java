package dp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author lilei
 * 递归解决01背包
 * 时间复杂度：2^n (n:物品数量)
 **/
public class RecursionStrategy implements Strategy {
    List<Integer> list = new ArrayList<>();

    @Override
    public int getMaxValue(int cap, int[] weight, int[] value) {
        sum(0, 0, 0, weight, value, cap);
        Collections.sort(list);
        return list.get(list.size() - 1);
    }

    private void sum(int totalValue, int totalWeight,
                     int height,
                     int[] weight, int[] value,
                     int cap) {

        if (totalWeight > cap) {
            return;
        }
        if (height >= weight.length) {
            //at bottom
            list.add(totalValue);
            return;
        }

        //choose
        sum(totalValue + value[height],
                totalWeight + weight[height],
                height + 1,
                weight, value, cap);

        //no choose
        sum(totalValue,
                totalWeight,
                height + 1,
                weight, value, cap);
    }
}