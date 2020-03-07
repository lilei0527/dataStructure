package tree;



/**
 * @author lilei
 * create at 2020/3/7 11:49
 * 二叉树
 */
@SuppressWarnings("unused")
public abstract class TwoForkTree<K extends Comparable<K>, V> implements Tree<K, V> {
    public TreeNode<K, V> root;

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

    protected TreeNode<K, V> successor(TreeNode<K,V> t) {
        if (t == null)
            return null;
        else if (t.right != null) {
            TreeNode<K,V> p = t.right;
            while (p.left != null)
                p = p.left;
            return p;
        } else {
            TreeNode<K,V> p = t.parent;
            TreeNode<K,V> ch = t;
            while (p != null && ch == p.right) {
                ch = p;
                p = p.parent;
            }
            return p;
        }
    }

    /**
     * Returns the predecessor of the specified Entry, or null if no such.
     */
    protected TreeNode<K, V> predecessor(TreeNode<K,V> t) {
        if (t == null)
            return null;
        else if (t.left != null) {
            TreeNode<K,V> p = t.left;
            while (p.right != null)
                p = p.right;
            return p;
        } else {
            TreeNode<K,V> p = t.parent;
            TreeNode<K,V> ch = t;
            while (p != null && ch == p.left) {
                ch = p;
                p = p.parent;
            }
            return p;
        }
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
