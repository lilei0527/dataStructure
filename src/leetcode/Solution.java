package leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author lilei
 **/
public class Solution {


    static class Ip {
        private List<String> ips = new ArrayList<>();
        private StringBuilder ip;

        /**
         * @author lilei
         * create at 13:55 2020/8/7
         * 给定ip字符串返回所有可能的ip地址
         * 比如给定25525512122，可能的ip为255.255.12.122或者255.255.121.22
         **/
        public List<String> getIps(String ip) {
            this.ip = new StringBuilder(ip);
            addIp(0,0);
            return ips;
        }


        /**
         * @param count:第几个点
         * @param index:点的下标
         **/
        private void addIp(int count, int index) {

            if (count > 2) {
                ips.add(ip.toString());
                return;
            }

            char nextChar = ip.charAt(index);

            //  x.x.x.a**  a>2
            if (count == 2 && nextChar > '2' && ip.length() - index >= 3) {
                return;
            }

            //only allow a** a<3
            int max = nextChar > '2' ? 2 : 3;

            for (int i = 0; i < max; i++) {
                ip.insert(index + i + 1, ".");
                addIp(count + 1, index + i + 1);
            }

        }
    }


    public static void main(String[] args) {
        Ip ip= new Ip();
        List<String> ips = ip.getIps("12312323232");
        System.out.println(ips);
    }
}
