package tree;


import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author lilei
 * create at 2020/3/7 11:49
 * 二叉树
 */
@SuppressWarnings("unused")
public abstract class TwoForkTree<K extends Comparable<K>, V> implements Tree<K, V> {
    public TreeNode<K, V> root;


    // 用于获得树的层数
    public int getTreeDepth(TreeNode<K, V> root) {
        return root == null ? 0 : (1 + Math.max(getTreeDepth(root.left), getTreeDepth(root.right)));
    }


    private void writeArray(TreeNode<K, V> currNode, int rowIndex, int columnIndex, String[][] res, int treeDepth) {
        // 保证输入的树不为空
        if (currNode == null) return;
        // 先将当前节点保存到二维数组中       ·hf'
        res[rowIndex][columnIndex] = String.valueOf(currNode.entry.key);

        // 计算当前位于树的第几层
        int currLevel = ((rowIndex + 1) / 2);
        // 若到了最后一层，则返回
        if (currLevel == treeDepth) return;
        // 计算当前行到下一行，每个元素之间的间隔（下一行的列索引与当前元素的列索引之间的间隔）
        int gap = treeDepth - currLevel - 1;

        // 对左儿子进行判断，若有左儿子，则记录相应的"/"与左儿子的值
        if (currNode.left != null) {
            res[rowIndex + 1][columnIndex - gap] = "/";
            writeArray(currNode.left, rowIndex + 2, columnIndex - gap * 2, res, treeDepth);
        }

        // 对右儿子进行判断，若有右儿子，则记录相应的"\"与右儿子的值
        if (currNode.right != null) {
            res[rowIndex + 1][columnIndex + gap] = "\\";
            writeArray(currNode.right, rowIndex + 2, columnIndex + gap * 2, res, treeDepth);
        }
    }


    public void print() {
        if (root == null) System.out.println("EMPTY!");
        // 得到树的深度
        int treeDepth = getTreeDepth(root);

        // 最后一行的宽度为2的（n - 1）次方乘3，再加1
        // 作为整个二维数组的宽度
        int arrayHeight = treeDepth * 2 - 1;
        int arrayWidth = (2 << (treeDepth - 2)) * 3 + 1;
        // 用一个字符串数组来存储每个位置应显示的元素
        String[][] res = new String[arrayHeight][arrayWidth];
        // 对数组进行初始化，默认为一个空格
        for (int i = 0; i < arrayHeight; i++) {
            for (int j = 0; j < arrayWidth; j++) {
                res[i][j] = " ";
            }
        }

        // 从根节点开始，递归处理整个树
        // res[0][(arrayWidth + 1)/ 2] = (char)(root.val + '0');
        writeArray(root, 0, arrayWidth / 2, res, treeDepth);

        // 此时，已经将所有需要显示的元素储存到了二维数组中，将其拼接并打印即可
        for (String[] line : res) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < line.length; i++) {
                sb.append(line[i]);
                if (line[i].length() > 1 && i <= line.length - 1) {
                    i += line[i].length() > 4 ? 2 : line[i].length() - 1;
                }
            }
            System.out.println(sb);
        }
    }


    public TreeNode<K, V> search(TreeNode<K, V> node, K key) {
        if (node == null) {
            return null;
        } else {
            if (isLeft(key, node)) {
                return search(node.left, key);
            }
            if (isRight(key, node)) {
                return search(node.right, key);
            }
            return node;
        }
    }

    public V search(K key) {
        TreeNode<K, V> search = search(root, key);
        return search == null ? null : search.entry.value;
    }

    protected TreeNode<K, V> findMax(TreeNode<K, V> treeNode) {
        if (treeNode == null) {
            return null;
        }
        if (treeNode.right != null) {
            return findMax(treeNode.right);
        } else {
            return treeNode;
        }
    }

    protected TreeNode<K, V> findMin(TreeNode<K, V> treeNode) {
        if (treeNode == null) {
            return null;
        }
        if (treeNode.left != null) {
            return findMin(treeNode.left);
        } else {
            return treeNode;
        }
    }

    //广度优先遍历
    public List<Entry<K, V>> bfsScan(){
        List<Entry<K,V>> result = new ArrayList<>();
        if(root==null){
            return new ArrayList<>();
        }
        bfsScan(Collections.singletonList(root),result);
        return result;
    }

    public void bfsScan(List<TreeNode<K, V>> nodes, List<Entry<K, V>> result) {
        if (nodes.isEmpty()) {
            return;
        }
        List<TreeNode<K, V>> child = new ArrayList<>();
        for (TreeNode<K, V> node : nodes) {
            result.add(node.entry);
            if (node.left != null) {
                child.add(node.left);
            }
            if (node.right != null) {
                child.add(node.right);
            }
        }
        bfsScan(child, result);
    }

    public List<Entry<K, V>> dfsScan(DFSType type){
        List<Entry<K,V>> result = new ArrayList<>();
        if(root==null){
            return new ArrayList<>();
        }
        switch (type){
            case FRONT:dfsScanFront(root,result);
            case MIDDLE:dfsScanMiddle(root,result);
            case BEHIND:dfsScanBehind(root,result);
        }
        return result;
    }

    //深度优先-前序遍历
    public  void dfsScanFront(TreeNode<K, V> node,List<Entry<K, V>> result){
        result.add(node.entry);
        if(node.left!=null){
            dfsScanFront(node.left,result);
        }
        if(node.right!=null){
            dfsScanFront(node.right,result);
        }
    }

    //深度优先-中序遍历
    public  void dfsScanMiddle(TreeNode<K, V> node,List<Entry<K, V>> result){

        if(node.left!=null){
            dfsScanMiddle(node.left,result);
        }

        result.add(node.entry);

        if(node.right!=null){
            dfsScanMiddle(node.right,result);
        }
    }

    //深度优先-后序遍历
    public  void dfsScanBehind(TreeNode<K, V> node,List<Entry<K, V>> result){
        if(node.left!=null){
            dfsScanBehind(node.left,result);
        }
        if(node.right!=null){
            dfsScanBehind(node.right,result);
        }
        result.add(node.entry);
    }

    K getKey(TreeNode<K, V> treeNode) {
        return treeNode.entry.key;
    }

    protected boolean hasLeft(TreeNode<K, V> treeNode) {
        return treeNode.left != null;
    }


    protected boolean hasRight(TreeNode<K, V> treeNode) {
        return treeNode.left != null || treeNode.right != null;
    }

    protected boolean hasChild(TreeNode<K, V> treeNode) {
        return treeNode.left != null || treeNode.right != null;
    }


    protected boolean isLeft(Entry<K, V> entry, TreeNode<K, V> treeNode) {
        return isLeft(entry.getKey(), treeNode);
    }

    protected boolean isLeft(K key, TreeNode<K, V> treeNode) {
        return key.compareTo(getKey(treeNode)) < 0;
    }

    protected boolean isRight(Entry<K, V> entry, TreeNode<K, V> treeNode) {
        return isRight(entry.getKey(), treeNode);
    }

    protected boolean isRight(K key, TreeNode<K, V> treeNode) {
        return key.compareTo(getKey(treeNode)) > 0;
    }

    protected boolean isSame(Entry<K, V> entry, TreeNode<K, V> treeNode) {
        return isSame(entry.getKey(), treeNode);
    }

    protected boolean isSame(K key, TreeNode<K, V> treeNode) {
        return key.compareTo(getKey(treeNode)) == 0;
    }

}
