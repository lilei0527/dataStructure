package leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author lilei
 **/
public class Solution {


    static class Ip {
        private List<String> ips = new ArrayList<>();
        private String ip;
        private StringBuilder sb = new StringBuilder();

        /**
         * @author lilei
         * create at 13:55 2020/8/7
         * 给定ip字符串返回所有可能的ip地址
         * 比如给定25525512122，可能的ip为255.255.12.122或者255.255.121.22
         **/
        public List<String> getIps(String ip) {
            this.ip = ip;
            addIp(0, 0);
            return ips;
        }


        /**
         * @param count:第几个点
         * @param index:点的下标
         **/
        private void addIp(int count, int index) {

            int last = ip.length() - index;

            if (last > (4 - count) * 3 || last < (4 - count)) {
                return;
            }

            char nextChar = ip.charAt(index);

            //last segment should not a** (a>2)
            if (count == 3
                    && getLastSegment(ip, index) <= 255
                    && !(nextChar == '0' && last > 1)) {

                sb.append(ip.substring(index));
                ips.add(sb.toString());
                return;
            }

            int max = nextChar == '0' ? 1 : nextChar > '2' ? 2 : 3;

            for (int i = 0; i < max && index + i + 1 < ip.length(); i++) {

                String substring = ip.substring(index, index + i + 1);
                if (getLastSegment(substring, 0) > 255) {
                    continue;
                }

                sb.append(substring);
                sb.append(".");
                addIp(count + 1, index + i + 1);
                sb.delete(index + count, sb.length());
            }
        }

        private int getLastSegment(String ip, int index) {
            int re = 0;
            for (; index < ip.length(); index++) {
                re = re * 10 + ip.charAt(index) - '0';
            }
            return re;
        }
    }


    public static void main(String[] args) {
        Ip ip = new Ip();
        List<String> ips = ip.getIps("21226991");
        System.out.println(ips);
    }
}
