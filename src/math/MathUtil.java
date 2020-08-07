package math;

/**
 * @author lilei
 **/
public class MathUtil {
    public static double pow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        return n > 0 ? quickMul(x, n) : 1 / quickMul(x, -n);
    }

    private static double quickMul(double x, int n) {
        double y = 1.0;
        while (n > 0) {
            if (n % 2 == 1) {
                y = y * x;
            }
            x *= x;
            n /= 2;
        }
        return y;
    }

    public static void main(String[] args) {
        System.out.println(MathUtil.pow(2, -2));
    }
}
