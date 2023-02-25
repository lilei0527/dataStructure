package leetcode;


import java.util.*;

/**
 * @author lilei
 **/
@SuppressWarnings("unused")
public class Solution {


    /**
     * 幂集。编写一种方法，返回某集合的所有子集。集合中不包含重复的元素。
     *  输入： nums = [1,2,3]
     *  输出：
     *  [3],
     *  [1],
     *  [2],
     *  [1,2,3],
     *  [1,3],
     *  [2,3],
     *  [1,2],
     *  []
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        subsets(nums,0,list,result);
        return result;
    }

    public void subsets(int[] nums,int index,List<Integer> list,List<List<Integer>> result){
        if(index == nums.length){
            result.add(new ArrayList<>(list));
            return;
        }

        //选择index位置上的数
        list.add(nums[index]);
        subsets(nums,index+1,list,result);

        //不选择index位置上的数
        list.remove(list.size()-1);
        subsets(nums,index+1,list,result);

    }


    static class NQueens {

        public List<Map<Integer,Integer>> getNQueens(int n) {
            List<Map<Integer,Integer>>list = new ArrayList<>();
            int[] segment = new int[n];
            fill(0, n, segment,list);
            return list;
        }

        private void fill(int row, int n, int[] segment,List<Map<Integer,Integer>>list) {
            if (row == n) {
                Map<Integer,Integer>queens = new TreeMap<>();
                for (int i = 0; i < segment.length; i++) {
                    queens.put(i,segment[i]);
                }
                list.add(queens);
                return;
            }

            int index = 0;
            int[] unused = new int[n - row];
            for (int i = 0; i < n; i++) {
                boolean flag = true;
                for (int j = 0; j < row; j++) {
                    if (i == segment[j]) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    unused[index++] = i;
                }
            }

            for (int i : unused) {
                segment[row] = i;
                fill(row + 1, n, segment,list);
            }
        }

        @SuppressWarnings("SameParameterValue")
        private void print(int n){
            List<Map<Integer, Integer>> nQueens = getNQueens(n);
            for(Map<Integer,Integer>map:nQueens) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if(j==map.get(i)){
                            System.out.print("Q");
                        }else {
                            System.out.print(" ");
                        }
                        if (j == n - 1) {
                            System.out.println();
                        }
                    }
                }
                System.out.println("--------------------");
            }
        }
    }

    static class Shape {
        public void diamond(int sideLength, boolean isCross) {
            for (int row = 0; row < 2 * sideLength - 1; row++) {
                for (int col = 0; col < 2 * sideLength - 1; col++) {
                    if (row < sideLength) {
                        if (col == sideLength - row - 1 ||
                                col == sideLength + row - 1 ||
                                (isCross && row > sideLength / 2 &&
                                        (col == row || col == 2 * sideLength - row - 2))) {
                            System.out.print("*");
                        } else {
                            System.out.print(" ");
                        }
                    } else {
                        if (col == row - sideLength + 1 ||
                                col == 3 * sideLength - row - 3 ||
                                (isCross && row < 1.5 * (sideLength - 1) &&
                                        (col == 2 * sideLength - row - 2 || col == row))) {
                            System.out.print("*");
                        } else {
                            System.out.print(" ");
                        }
                    }
                }
                System.out.println();
            }
        }
    }

    static class ListUtil {
        static class ListNode {
            int val;
            ListNode next = null;

            ListNode(int val) {
                this.val = val;
            }

            @Override
            public String toString() {
                return "ListNode{" +
                        "val=" + val +
                        ", next=" + next +
                        '}';
            }
        }

        public static ListNode ReverseList(ListNode head) {
            if (head == null) return null;
            ListNode pre = null, next;
            while (head.next != null) {
                next = head.next;
                head.next = pre;
                pre = head;
                head = next;
            }
            head.next = pre;
            return head;
        }

        public static boolean hasCycle(ListNode head) {
            ListNode fast = head;


            while (fast != null && fast.next != null) {
                fast = fast.next.next;
                head = head.next;
                if (fast == head) {
                    return true;
                }
            }
            return false;
        }

        //将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的
        public static ListNode mergeList(ListNode list1,ListNode list2){
            ListNode listNode = new ListNode(-1);
            ListNode head = listNode;
            while (list1!=null&&list2!=null){
                if(list1.val>=list2.val){
                    head.next = list2;
                    list2 = list2.next;
                }else {
                    head.next = list1;
                    list1 = list1.next;
                }
                head = head.next;
            }

            if(list1==null){
                head.next = list2;
            }else {
                head.next = list1;
            }
            return listNode.next;
        }
    }

    static class Permutation {
        /**
         * @author lilei
         * create at 2020/8/15 17:20
         * 给定一个字符序列，返回所有不同序列的情况
         */
        public <E> Set<List<E>> rank(E[] es) {
            Set<List<E>> all = new HashSet<>();
            boolean[] used = new boolean[es.length];
            int[] signal = new int[es.length];
            dfs(0, all, signal, es, used);
            return all;
        }

        private <E> void dfs(int deep, Set<List<E>> all,
                             int[] signal, E[] es, boolean[] used) {
            if (deep == es.length) {
                List<E> list = new ArrayList<>();
                for (int i = 0; i < es.length; i++) {
                    list.add(es[signal[i]]);
                }
                all.add(list);
                return;
            }

            for (int i = 0; i < es.length; i++) {
                if (used[i]) {
                    continue;
                }
                signal[deep] = i;
                used[i] = true;
                dfs(deep + 1, all, signal, es, used);
                used[i] = false;
            }
        }



        public List<List<Integer>> permute(Integer[] nums) {
            List<Integer> ints = Arrays.asList(nums);
            List<List<Integer>> result= new ArrayList<>();
            backtrack(nums.length,ints,result,0);
            return result;
        }

        private void backtrack(int length,List<Integer>list,List<List<Integer>> result,int index){
            if(index==length){
                result.add(new ArrayList<>(list));
                return;
            }

            for (int i = index; i < length; i++) {
                Collections.swap(list,index,i);
                backtrack(length,list,result,index+1);
                Collections.swap(list,index,i);
            }
        }
    }

    static class Ip {
        private final List<String> ips = new ArrayList<>();
        private String ip;
        private final int[] segment = new int[3];

        /**
         * @author lilei
         * create at 13:55 2020/8/7
         * 给定ip字符串返回所有可能的ip地址
         * 比如给定25525512122，可能的ip为255.255.12.122或者255.255.121.22
         **/
        public List<String> getIps(String ip) {
            if (isNotNumber(ip)) return ips;
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

        private boolean isNotNumber(String s) {
            char[] chars = s.toCharArray();
            for (char c : chars) {
                if (c == '.') continue;
                if (c <= '0' || c >= '9') {
                    return true;
                }
            }
            return false;
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
                if (isNotNumber(s)) return false;
            }
            return true;
        }
    }

    static class MedianFinder {
        private final PriorityQueue<Double> maxheap = new PriorityQueue<>((x, y) -> (int) (y - x));
        private final PriorityQueue<Double> minheap = new PriorityQueue<>();

        /**
         * initialize your data structure here.
         */
        public MedianFinder() {
        }

        public void addNum(double num) {
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

        public double findMedian(List<Double> list) {
            for (Double d : list) {
                addNum(d);
            }
            return findMedian();
        }


        public double findMedian() {
            Double max = maxheap.peek();
            Double min = minheap.peek();
            max = max == null ? 0 : max;
            min = min == null ? 0 : min;
            if (maxheap.size() == minheap.size()) {
                return (max + min) / 2;
            }
            if (maxheap.size() > minheap.size()) {
                return max;
            }
            return min;
        }

    }

    static class Counter {
        /**
         * @author lilei
         * create at 11:54 2020/8/11
         * 前K个出现频率最高的元素
         **/
        public <E> List<E> topK(E[] element, int k) {

            Map<E, Integer> count = new HashMap<>();

            for (E e : element) {
                count.merge(e, 1, Integer::sum);
            }

            PriorityQueue<E> maxheap =
                    new PriorityQueue<>(((o1, o2) -> count.get(o2) - count.get(o1)));

            for (E e : count.keySet()) {
                maxheap.offer(e);
            }

            List<E> es = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                E poll = maxheap.poll();
                if (poll == null) {
                    break;
                }
                es.add(poll);
            }

            return es;
        }


        /**
         * @author lilei
         * create at 10:48 2020/7/28
         * 求给定数组连续子数组的最大值（dp求解）
         * 时间复杂度O(n) 空间复杂度O(1)
         **/
        public int maxSubArray(int[] nums) {
            int res = nums[0];
            int max = 0;
            for (int i = 1; i < nums.length; i++) {
                max = Math.max(max + nums[i], 0);
                res = Math.max(max, res);
            }
            return res;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 最大连续e的个数,e如果为null，返回此数组最大连续元素的个数
         **/
        public <E> int maxRepeat(E[] es, E e) {
            int max = 0;
            int p = 0;
            while (p < es.length) {
                if (es[p] != e && e != null) {
                    p++;
                    continue;
                }
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                max = Math.max(step, max);
                p = pn;
            }
            return max;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 返回数组es最大连续相同元素
         **/
        public <E> List<E> maxRepeat(E[] es) {
            int max = 0;
            int p = 0;
            List<E> maxEs = new ArrayList<>();
            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                if (max == step) {
                    maxEs.add(es[p]);
                } else {
                    max = Math.max(step, max);
                    if (max == step) {
                        maxEs.clear();
                        maxEs.add(es[p]);
                    }
                }
                p = pn;
            }
            return maxEs;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 返回数组es最大连续相同元素与元素的个数
         **/
        public <E> Map.Entry<List<E>, Integer> maxRepeatAndCount(E[] es) {
            int max = 0;
            int p = 0;
            List<E> maxEs = new ArrayList<>();
            Map.Entry<List<E>, Integer> maxEsIn = new HashMap.SimpleEntry<>(maxEs, 0);

            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                if (max == step) {
                    maxEs.add(es[p]);
                } else {
                    max = Math.max(step, max);
                    if (max == step) {
                        maxEs.clear();
                        maxEs.add(es[p]);
                    }
                }
                p = pn;
            }
            maxEsIn.setValue(max);
            return maxEsIn;
        }

        /**
         * @author lilei
         * create at 14:25 2020/8/11
         * 返回数组es连续相同元素与元素的个数(同元素取最大个数)
         **/
        public <E> Map<Integer, List<E>> repeatAndCount(E[] es) {
            int p = 0;
            Map<E, Integer> countMap = new HashMap<>();

            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                int step = pn - p;
                Integer count = countMap.get(es[p]);
                if (count == null) {
                    countMap.put(es[p], step);
                } else {
                    if (step > count) {
                        countMap.put(es[p], step);
                    }
                }
                p = pn;
            }
            return mergeSameValue(countMap);
        }

        private <E> List<E> removeRepeat(E[] es) {
            List<E> list = new ArrayList<>();
            int p = 0;
            while (p < es.length) {
                int pn = p + 1;
                while (pn < es.length && es[p].equals(es[pn])) {
                    pn++;
                }
                list.add(es[p]);
                p = pn;
            }
            return list;
        }

        private String removeDuplicate(String s) {
            Stack<Character> stack = new Stack<>();
            HashSet<Character> seen = new HashSet<>();
            HashMap<Character, Integer> last_occurrence = new HashMap<>();
            for (int i = 0; i < s.length(); i++) last_occurrence.put(s.charAt(i), i);
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (!seen.contains(c)) {
                    while (!stack.isEmpty() && c < stack.peek() && last_occurrence.get(stack.peek()) > i) {
                        seen.remove(stack.pop());
                    }
                    seen.add(c);
                    stack.push(c);
                }
            }
            StringBuilder sb = new StringBuilder(stack.size());
            for (Character c : stack) sb.append(c.charValue());
            return sb.toString();
        }

        private <E> Map<Integer, List<E>> mergeSameValue(Map<E, Integer> map) {
            Map<Integer, List<E>> reMap = new HashMap<>();
            for (Map.Entry<E, Integer> e : map.entrySet()) {
                Integer value = e.getValue();
                List<E> es = reMap.get(value);
                if (es == null) {
                    es = new ArrayList<>();
                }
                es.add(e.getKey());
                reMap.put(e.getValue(), es);
            }
            return reMap;
        }
    }

    /**
     请你按照从括号内到外的顺序，逐层反转每对匹配括号中的字符串，并返回最终的结果。

     注意，您的结果中 不应 包含任何括号。
     示例 1：

     输入：s = “(abcd)”
     输出：“dcba”

     示例 2：

     输入：s = “(u(love)i)”
     输出：“iloveu”

     示例 3：

     输入：s = “(ed(et(oc))el)”
     输出：“leetcode”
     */
    public static class StringUtil{
        public String reverseBracketStr(String s){
            LinkedList<String> stack = new LinkedList<>();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < s.length(); i++) {
                char ch = s.charAt(i);
                if (ch == '(') {
                    stack.push(sb.toString());
                    sb.setLength(0);
                } else if (ch == ')') {
                    sb.reverse();
                    sb.insert(0, stack.pop());
                } else {
                    sb.append(ch);
                }
            }
            return sb.toString();
        }

        public List<String> generateParenthesis(int n) {
            List<String> ans = new ArrayList<>();
            backtrack(ans, new StringBuilder(), 0, 0, n);
            return ans;
        }

        public void backtrack(List<String> ans, StringBuilder cur, int open, int close, int max) {
            if (cur.length() == max * 2) {
                ans.add(cur.toString());
                return;
            }
            if (open < max) {
                cur.append('(');
                backtrack(ans, cur, open + 1, close, max);
                cur.deleteCharAt(cur.length() - 1);
            }
            if (close < open) {
                cur.append(')');
                backtrack(ans, cur, open, close + 1, max);
                cur.deleteCharAt(cur.length() - 1);
            }
        }

    }


    public static void main(String[] args) {
//        Ip ip = new Ip();
//        List<String> ips = ip.getIps("1231232123");
//        boolean b = ip.validIp4("12.23.34.12");
//        System.out.println(b);
//        System.out.println(ips);
//
//
//        MedianFinder medianFinder = new MedianFinder();
//        medianFinder.addNum(1);
//        medianFinder.addNum(9);
//        medianFinder.addNum(3);
//        medianFinder.addNum(2);
//        medianFinder.addNum(2);
//        System.out.println("中位数" + medianFinder.findMedian());
//
//
//        Counter counter = new Counter();
//        Integer[] element = {1, 22, 22, 22, 22, 1, 1, 22, 14, 12, 12, 12, 12};
//        List<Integer> integers = counter.topK(element, 10);
//        System.out.println("字符前K个频率最高" + integers);
//
//        int[] nums = {-1, 2, 1, -1, 2, -1, 2};
//        System.out.println("最大连续子数组" + counter.maxSubArray(nums));
//        System.out.println("查找最大连续元素个数" + counter.maxRepeat(element, 123));
//        System.out.println("最大连续元素" + counter.maxRepeat(element));
//        System.out.println("最大连续元素和个数" + counter.maxRepeatAndCount(element));
//        System.out.println("所有连续元素信息" + counter.repeatAndCount(element));
//        System.out.println("移除连续元素后" + counter.removeRepeat(element));
//
//        Permutation permutation = new Permutation();
//        Integer[] integers1 = {1, 2, 3};
//        Set<List<Integer>> rank = permutation.rank(integers1);
//        System.out.println(rank);
//        System.out.println(rank.size());
//
//
//        String s = "bcabvxcjpwrtup123213sfs";
//        String s1 = counter.removeDuplicate(s);
//        System.out.println(s1);
//
//        Shape shape = new Shape();
//        shape.diamond(11, true);
//
//        NQueens nQueens = new NQueens();
//        nQueens.print(10);
//        StringUtil stringUtil = new StringUtil();
//        String s = stringUtil.reverseBracketStr("(ed(et(oc)(od))el)");
//        System.out.println(s);

//        ListUtil.ListNode listNode1 = new ListUtil.ListNode(1);
//        ListUtil.ListNode listNode3 = new ListUtil.ListNode(3);
//        ListUtil.ListNode listNode5 = new ListUtil.ListNode(5);
//        listNode1.next = listNode3;
//        listNode3.next = listNode5;
//
//        ListUtil.ListNode listNode2 = new ListUtil.ListNode(2);
//        ListUtil.ListNode listNode4 = new ListUtil.ListNode(4);
//        ListUtil.ListNode listNode6 = new ListUtil.ListNode(6);
//        listNode2.next = listNode4;
//        listNode4.next = listNode6;
//
//        ListUtil.ListNode listNode = ListUtil.mergeList(listNode1, listNode2);
//        System.out.println(listNode);

//        Permutation permutation = new Permutation();
        int[] temp = new int[]{1,2,3};
//        List<List<Integer>> permute = permutation.permute(temp);
//        System.out.println(permute);

        Solution solution = new Solution();
        List<List<Integer>> subsets = solution.subsets(temp);
        System.out.println(subsets);
    }
}
