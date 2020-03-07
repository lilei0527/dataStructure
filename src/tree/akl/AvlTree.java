package tree.akl;

import tree.Entry;
import tree.Tree;

@SuppressWarnings("unused")
public class AvlTree implements Tree {
    private Node root;

    @Override
    public void insert(Entry entry) {
        root = insert(entry, root);
    }

    public Node insert(Entry entry, Node node) {
        if (node == null) {
            node = new Node(entry);
        } else {
            if (isLeft(entry, node)) {
                node.left = insert(entry, node.left);
                node = rotate(node, entry);
            }

            if (isRight(entry, node)) {
                node.right = insert(entry, node.right);
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
    public Entry search(int key) {
        return search(root, key);
    }

    public Entry search(Node node, int key) {
        if (node == null) {
            return null;
        } else {
            if (isLeft(key, node)) {
                return search(node.left, key);
            }
            if (isRight(key, node)) {
                return search(node.right, key);
            }
            return node.entry;
        }
    }


    @Override
    public Entry delete(int key) {
        root = delete(root, key);
        return search(root, key);
    }

    public Node delete(Node node, int key) {
        if (node == null) {
            return null;
        } else {

            if (isSame(key, node)) {

                if (!hasChild(node)) {
                    return null;
                }

                //左子树的高度大于等于右子树
                if (height(node.left) >= height(node.right)) {
                    //从左子树中选择一个最大值来代替被删除节点
                    int max = findMax(node);
                    Entry entry = node.entry;
                    entry.setKey(max);
                    node.entry = entry;
                    delete(node.left, max);
                } else {
                    int min = findMin(node);
                    Entry entry = node.entry;
                    entry.setKey(min);
                    node.entry = entry;
                    delete(node.right, min);
                }
            }


            if (isLeft(key, node)) {
                node.left = delete(node.left,key);

                if (height(node.right) - height(node.left) == 2) {
                    if (height(node.right == null ? null : node.right.right)
                            >= height(node.right == null ? null : node.right.left)) {
                        //RR
                        node = leftRotate(node);
                    } else {
                        //RL
                        node = rightLeftRotate(node);
                    }
                }
            }


            if (isRight(key, node)) {
                node.right = delete(node.right,key);
                if (height(node.left) - height(node.right) == 2) {
                    if (height(node.right == null ? null : node.right.left)
                            >= height(node.right == null ? null : node.right.right)) {
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


    private Node rotate(Node node, int key) {
        if (height(node.left) - height(node.right) == 2) {
            if (isLeft(key, node.left)) {
                //LL
                node = rightRotate(node);
            } else {
                //LR
                node = leftRightRotate(node);
            }
        } else if (height(node.right) - height(node.left) == 2) {
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

    private Node rotate(Node node, Entry entry) {
        return rotate(node, entry.getKey());
    }


    private int findMax(Node node) {
        if (node == null) {
            return 0;
        }
        if (node.right != null) {
            return findMax(node);
        } else {
            return getKey(node);
        }
    }

    private int findMin(Node node) {
        if (node == null) {
            return 0;
        }
        if (node.left != null) {
            return findMin(node);
        } else {
            return getKey(node);
        }
    }

    //左旋
    private Node leftRotate(Node node) {
        Node right = node.right;
        node.right = right.left;
        right.left = node;

        //更新node和node.right的高度
        updateHeight(node);
        updateHeight(right);

        return right;
    }

    //右旋
    private Node rightRotate(Node node) {
        Node left = node.left;
        node.left = left.right;
        left.right = node;


        //更新node和node.left的高度
        updateHeight(node);
        updateHeight(left);
        return left;
    }

    //更新根节点的高度
    private void updateHeight(Node node) {
        node.height = Math.max(height(node.left),height(node.right)+1);
    }

    //判断是否平衡
    private boolean isBalance(Node node) {
        return Math.abs(height(node.left) - height(node.right)) < 2;
    }

    //先左旋后右旋
    private Node leftRightRotate(Node node) {
        node.left = leftRotate(node.left);
        return rightRotate(node);
    }

    //先右旋后左旋
    private Node rightLeftRotate(Node node) {
        node.right = rightRotate(node.right);
        return leftRotate(node);
    }


    public int height(Node node) {
        if (node == null) {
            return 0;
        } else {
            return node.height;
        }

    }

    private boolean hasLeft(Node node) {
        return node.left != null;
    }


    private boolean hasRight(Node node) {
        return node.left != null || node.right != null;
    }

    private boolean hasChild(Node node) {
        return node.left != null || node.right != null;
    }

    private int getKey(Node node) {
        return node.entry.key;
    }

    public boolean hasRoot() {
        return root != null;
    }

    private boolean isLeft(Entry entry, Node node) {
        return isLeft(entry.getKey(), node);
    }

    private boolean isLeft(int key, Node node) {
        return key < getKey(node);
    }

    private boolean isRight(Entry entry, Node node) {
        return isRight(entry.getKey(), node);
    }

    private boolean isRight(int key, Node node) {
        return key > getKey(node);
    }

    private boolean isSame(Entry entry, Node node) {
        return isSame(entry.getKey(), node);
    }

    private boolean isSame(int key, Node node) {
        return key == getKey(node);
    }


    private boolean isRoot(Node node) {
        return node == root;
    }


    public static void main(String[] args) {
        AvlTree tree = new AvlTree();
        tree.insert(new Entry(1, 1));
        tree.insert(new Entry(4, 4));
        tree.insert(new Entry(8, 8));
        tree.insert(new Entry(3, 3));
        tree.insert(new Entry(3, 3));
        tree.insert(new Entry(2, 2));
        System.out.println(tree.root);
        System.out.println(tree.search(4).getKey());
        tree.delete(8);
        System.out.println(tree.root);
    }
}
