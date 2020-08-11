package leetcode;


import java.util.*;

/**
 * @author lilei
 **/
public class Solution {


    static class Ip {
        private List<String> ips = new ArrayList<>();
        private String ip;
        private int[] segment = new int[3];

        /**
         * @author lilei
         * create at 13:55 2020/8/7
         * 给定ip字符串返回所有可能的ip地址
         * 比如给定25525512122，可能的ip为255.255.12.122或者255.255.121.22
         **/
        public List<String> getIps(String ip) {
            if (!isNumber(ip)) return ips;
            this.ip = ip;
            addIp(0, 0, 0);
            return ips;
        }


        /**
         * @param count:第几个点
         * @param index:点的下标
         **/
        private void addIp(int count, int index, int height) {

            int last = ip.length() - index;

            if (last > (4 - count) * 3 || last < (4 - count)) {
                return;
            }

            char nextChar = ip.charAt(index);

            //last segment should not a** (a>2)
            if (count == 3
                    && getLastSegment(ip, index) <= 255
                    && !(nextChar == '0' && last > 1)) {
                ips.add(split());
                return;
            }

            int max = nextChar == '0' ? 1 : nextChar > '2' ? 2 : 3;

            for (int i = 0; i < max && index + i + 1 < ip.length(); i++) {

                String substring = ip.substring(index, index + i + 1);

                if (getLastSegment(substring, 0) > 255) {
                    continue;
                }


                if (height < 3)
                    segment[height] = index + i + 1;

                addIp(count + 1, index + i + 1, height + 1);
            }
        }

        private boolean isNumber(String s) {
            char[] chars = s.toCharArray();
            for (char c : chars) {
                if (c == '.') continue;
                if (c <= '0' || c >= '9') {
                    return false;
                }
            }
            return true;
        }

        private int getLastSegment(String ip, int index) {
            int re = 0;
            for (; index < ip.length(); index++) {
                re = re * 10 + ip.charAt(index) - '0';
            }
            return re;
        }

        private String split() {
            StringBuilder stringBuilder = new StringBuilder(ip);
            stringBuilder.insert(segment[0], ".");
            stringBuilder.insert(segment[1] + 1, ".");
            stringBuilder.insert(segment[2] + 2, ".");
            return stringBuilder.toString();
        }

        public boolean validIp4(String ip) {
            String[] split = ip.split("\\.");
            for (String s : split) {
                if (s == null || s.length() == 0) return false;
                if (s.length() > 3) return false;
                if (s.length() != 1 && s.charAt(0) == '0') return false;
                if (getLastSegment(s, 0) > 255) return false;
                if (!isNumber(s)) return false;
            }
            return true;
        }
    }

    static class MedianFinder {
        private PriorityQueue<Integer> maxheap = new PriorityQueue<>((x, y) -> y - x);
        private PriorityQueue<Integer> minheap = new PriorityQueue<>();

        /**
         * initialize your data structure here.
         */
        public MedianFinder() {
        }

        public void addNum(int num) {
            if (minheap.isEmpty() || minheap.peek() <= num) {
                minheap.offer(num);
            } else {
                maxheap.offer(num);
            }
            if (minheap.size() - maxheap.size() == 2) {
                maxheap.offer(minheap.poll());
            }
            if (maxheap.size() - minheap.size() == 2) {
                minheap.offer(maxheap.poll());
            }
        }


        public double findMedian() {
            Integer max = maxheap.peek();
            Integer min = minheap.peek();
            max = max == null ? 0 : max;
            min = min == null ? 0 : min;
            if (maxheap.size() == minheap.size()) {
                return (double) (max + min) / 2;
            }
            if (maxheap.size() > minheap.size()) {
                return max;
            }
            return min;
        }
    }


    public static void main(String[] args) {
        Ip ip = new Ip();
        List<String> ips = ip.getIps("1231232123");
        boolean b = ip.validIp4("12.23.34.12");
        System.out.println(b);
        System.out.println(ips);

        MedianFinder medianFinder = new MedianFinder();
        medianFinder.addNum(1);
        medianFinder.addNum(9);
        medianFinder.addNum(3);
        medianFinder.addNum(2);
        medianFinder.addNum(2);
        System.out.println(medianFinder.findMedian());
    }
}
