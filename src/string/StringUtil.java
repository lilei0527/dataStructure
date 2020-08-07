package string;

/**
 * @author lilei
 **/
public class StringUtil {
    Comparable comparator = new Sunday();

    public boolean contains(char[] lcs, char[] scs) {
        return comparator.contains(lcs, scs);
    }



    public static void main(String[] args) {
        char[] lcs = {'h', 'e', 'r', 'e', ' ', 'i', 's', ' ', 'a', ' ', 's', 'i', 'm', 'p', 'l', 'e', ' ', 'e', 'x', 'a', 'm', 'p', 'l', 'e'};
        char[] scs = {'e', 'x', 'a', 'm', 'p', 'l', 'e'};
        StringUtil stringUtil = new StringUtil();
        boolean contains = stringUtil.contains(lcs, scs);
        System.out.println(contains);
    }
}
