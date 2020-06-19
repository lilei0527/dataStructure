package tree.akl;

import tree.Entry;
import tree.TreeNode;
import tree.TwoForkTree;

@SuppressWarnings("unused")
public class AvlTree<K extends Comparable<K>, V> extends TwoForkTree<K, V> {

    @Override
    public void insert(K k, V v) {
        Entry<K, V> entry = new Entry<>(k, v);
        root = insert(entry, (Node<K, V>) root);
    }

    public Node<K, V> insert(Entry<K, V> entry, Node<K, V> node) {
        if (node == null) {
            node = new Node<>(entry);


        } else {
            if (isLeft(entry, node)) {
                node.left = insert(entry, (Node<K, V>) node.left);
                node = rotate(node, entry);
            }

            if (isRight(entry, node)) {
                node.right = insert(entry, (Node<K, V>) node.right);
                node = rotate(node, entry);
            }

            if (isSame(entry, node)) {
                node.entry = entry;
            }
        }

        updateHeight(node);
        return node;
    }




    @Override
    public V delete(K key) {
        V value = search(key);
        root = delete((Node<K, V>) root, key);
        return value;
    }

    public Node<K, V> delete(Node<K, V> node, K key) {
        if (node == null) {
            return null;
        } else {

            if (isSame(key, node)) {

                if (!hasChild(node)) {
                    return null;
                }

                //左子树的高度大于等于右子树
                if (height((Node<K, V>) node.left) >= height((Node<K, V>) node.right)) {
                    //从左子树中选择一个最大值来代替被删除节点
                    Node<K, V> max = (Node<K, V>) findMax(node);
                    node.entry = max.entry;
                    delete((Node<K, V>) node.left, max.entry.key);
                } else {
                    TreeNode<K, V> min = findMin(node);
                    Entry<K, V> entry = node.entry;
                    entry.setKey(min.entry.key);
                    node.entry = entry;
                    delete((Node<K, V>) node.right, min.entry.key);
                }
            }


            if (isLeft(key, node)) {
                node.left = delete((Node<K, V>) node.left, key);

                if (height((Node<K, V>) node.right) - height((Node<K, V>) node.left) == 2) {
                    if (height(node.right == null ? null : (Node<K, V>) node.right.right)
                            >= height(node.right == null ? null : (Node<K, V>) node.right.left)) {
                        //RR
                        node = leftRotate(node);
                    } else {
                        //RL
                        node = rightLeftRotate(node);
                    }
                }
            }


            if (isRight(key, node)) {
                node.right = delete((Node<K, V>) node.right, key);
                if (height((Node<K, V>) node.left) - height((Node<K, V>) node.right) == 2) {
                    if (height(node.right == null ? null : (Node<K, V>) node.right.left)
                            >= height(node.right == null ? null : (Node<K, V>) node.right.right)) {
                        //LL
                        node = rightRotate(node);
                    } else {
                        //LR
                        node = leftRightRotate(node);
                    }
                }
            }

            return node;
        }
    }


    private Node<K, V> rotate(Node<K, V> node, K key) {
        if (height((Node<K, V>) node.left) - height((Node<K, V>) node.right) == 2) {
            if (isLeft(key, node.left)) {
                //LL
                node = rightRotate(node);
            } else {
                //LR
                node = leftRightRotate(node);
            }
        } else if (height((Node<K, V>) node.right) - height((Node<K, V>) node.left) == 2) {
            if (isRight(key, node.right)) {
                //RR
                node = leftRotate(node);
            } else {
                //RL
                node = rightLeftRotate(node);
            }
        }
        return node;
    }

    private Node<K, V> rotate(Node<K, V> node, Entry<K, V> entry) {
        return rotate(node, entry.getKey());
    }


    //左旋
    private Node<K, V> leftRotate(Node<K, V> node) {
        Node<K, V> right = (Node<K, V>) node.right;
        node.right = right.left;
        right.left = node;

        //更新node和node.right的高度
        updateHeight(node);
        updateHeight(right);

        return right;
    }

    //右旋
    private Node<K, V> rightRotate(Node<K, V> node) {
        Node<K, V> left = (Node<K, V>) node.left;
        node.left = left.right;
        left.right = node;


        //更新node和node.left的高度
        updateHeight(node);
        updateHeight(left);
        return left;
    }

    //更新根节点的高度
    private void updateHeight(Node<K, V> node) {
        node.height = Math.max(height((Node<K, V>) node.left), height((Node<K, V>) node.right)) + 1;
    }

    //判断是否平衡
    private boolean isBalance(Node<K, V> node) {
        return Math.abs(height((Node<K, V>) node.left) - height((Node<K, V>) node.right)) < 2;
    }

    //先左旋后右旋
    private Node<K, V> leftRightRotate(Node<K, V> node) {
        node.left = leftRotate((Node<K, V>) node.left);
        return rightRotate(node);
    }

    //先右旋后左旋
    private Node<K, V> rightLeftRotate(Node<K, V> node) {
        node.right = rightRotate((Node<K, V>) node.right);
        return leftRotate(node);
    }


    public int height(Node<K, V> node) {
        if (node == null) {
            return 0;
        } else {
            return node.height;
        }

    }



}
