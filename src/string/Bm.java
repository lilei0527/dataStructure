package string;

import java.util.HashMap;
import java.util.Map;

/**
 * @author lilei
 **/
public class Bm implements Comparable {
    @Override
    public boolean contains(char[] lcs, char[] scs) {
        Map<Character, Integer> badMap = createBad(scs);
        int[] fine = createFine(scs);

        int index = scs.length - 1;
        while (index < lcs.length) {
            int scsIndex = scs.length - 1;
            int tempIndex = index;
            while (true) {
                if (lcs[tempIndex] == scs[scsIndex]) {
                    tempIndex--;
                    scsIndex--;
                    if (scsIndex == 0) {
                        return true;
                    }
                } else {

                    char bad = lcs[tempIndex];
                    Integer bidIndex = badMap.get(bad);
                    bidIndex = bidIndex == null ? -1 : bidIndex;
                    int badStep = scsIndex - bidIndex;

                    int fineStep;
                    if (scsIndex == scs.length - 1) {
                        fineStep = 0;
                    } else {
                        fineStep = fine[scsIndex + 1]==0?scs.length:fine[scsIndex+1];
                    }

                    int step = Math.max(fineStep, badStep);
                    index += step;
                    break;
                }
            }
        }
        return false;
    }

    private Map<Character, Integer> createBad(char[] cs) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < cs.length; i++) {
            map.put(cs[i], i);
        }
        return map;
    }

    private int[] createFine(char[] cs) {
        int[] fine = new int[cs.length];

        int cursor = 1;
        int next = cs.length - 2;
        while (cursor <= cs.length) {
            int step = 0;
            for (int i = 0; i < cursor; i++) {
                if (next >= i && cs[cs.length - 1 - i] == cs[next - i]) {
                    step++;
                } else {
                    break;
                }
            }

            if (step == cursor) {
                fine[cs.length - cursor] = cs.length - next - 1;
                cursor++;
            } else {
                if (next <= cursor - 1) {
                    cursor++;
                } else {
                    next--;
                }
            }
        }

        return fine;
    }

}
