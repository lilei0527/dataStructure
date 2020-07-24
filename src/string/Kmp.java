package string;

/**
 * @author lilei
 **/
public class Kmp implements Comparable {
    //KMP算法
    public boolean contains(char[] lcs, char[] scs) {
        int[] next = createNext(scs);
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
                    lcsIndex += next[scsIndex];
                    break;
                }
            }
        }
        return false;
    }


    private int[] createNext(char[] chars) {
        int[] ints = new int[chars.length];
        int i = 0, j = 1;
        ints[0] = 1;
        while (j < chars.length) {
            if (chars[i] == chars[j]) {
                ints[j] = j - i;
                i++;
            } else {
                ints[j] = j + 1;
            }
            j++;
        }
        return ints;
    }
}
