package string;

/**
 * @author lilei
 **/
public class Bm implements Comparable {
    @Override
    public boolean contains(char[] lcs, char[] scs) {
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
                    int fine = -1;
                    for (int i = 0; i < scsIndex; i++) {
                        if (scs[i] == lcs[tempIndex]) {
                            fine = i;
                        }
                    }
                    int step = scsIndex - fine;

                    for (int i = scsIndex + 1; i < scs.length; i++) {
                        boolean isFind = false;
                        for (int j = 0; j < scsIndex; j++) {
                            if (scs[j] != scs[i]) {
                                break;
                            }
                            if (i == scs.length - 1 && scs[j] == scs[i]) {
                                isFind = true;
                                break;
                            }
                            isFind = true;
                        }
                        if (isFind) {
                            step = i;
                            break;
                        }
                    }
                    index += step;
                    break;
                }
            }
        }
        return false;
    }
}
