package string;

import java.util.HashMap;
import java.util.Map;

/**
 * @author lilei
 * 从后往前比较的字符串比较算法
 * 时间复杂度O(n)
 * 空间复杂度O(m)
 **/
public class Sunday implements Comparable {
    @Override
    public boolean contains(char[] lcs, char[] scs) {
        Map<Character, Integer> next = createNext(scs);
        int lcsIndex = 0;
        while (lcsIndex < lcs.length) {
            int scsIndex = 0;
            int tempIndex = lcsIndex;
            while (true) {
                if (lcs[tempIndex] == scs[scsIndex]) {
                    scsIndex++;
                    tempIndex++;
                    if (scsIndex == scs.length) {
                        return true;
                    }
                } else {
                    int n = tempIndex + (scs.length - scsIndex);
                    lcsIndex += next.get(lcs[n]) == null ? scs.length + 1 : next.get(lcs[n]);
                    break;
                }
            }
        }
        return false;
    }

    private Map<Character, Integer> createNext(char[] scs) {

        Map<Character, Integer> map = new HashMap<>();

        for (int i = 0; i < scs.length; i++) {
            map.put(scs[i], scs.length - i);
        }

        return map;
    }
}
