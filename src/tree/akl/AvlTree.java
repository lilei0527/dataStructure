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
                node.setLeft(insert(entry, node.getLeft()));
                if (height(node.getLeft()) - height(node.getRight()) == 2) {
                    if (isLeft(entry, node.getLeft())) {
                        //LL
                        node = rightRotate(node);
                    } else {
                        //LR
                        node = leftRightRotate(node);
                    }
                }
            }

            if (isRight(entry, node)) {
                node.setRight(insert(entry, node.getRight()));
                if (height(node.getRight()) - height(node.getLeft()) == 2) {
                    if (isRight(entry, node.getRight())) {
                        //RR
                        node = leftRotate(node);
                    } else {
                        //RL
                        node = rightLeftRotate(node);
                    }
                }
            }

            if (isSame(entry, node)) {
                node.setEntry(entry);
            }
        }
        updateHeight(node);
        return node;
    }

    @Override
    public Entry search(int key) {
        return search(key, root);
    }

    public Entry search(int key, Node node) {
        if (node == null) {
            return null;
        } else {
            if (isLeft(key, node)) {
                return search(key, node.getLeft());
            }
            if (isRight(key, node)) {
                return search(key, node.getRight());
            }
            return node.getEntry();
        }
    }


    @Override
    public boolean delete(int key) {
        return false;
    }

    //左旋
    private Node leftRotate(Node node) {
        Node right = node.getRight();
        node.setRight(right.getLeft());
        right.setLeft(node);

        //更新node和node.right的高度
        updateHeight(node);
        updateHeight(right);

        return right;
    }

    //右旋
    private Node rightRotate(Node node) {
        Node left = node.getLeft();
        node.setLeft(left.getRight());
        left.setRight(node);

        //更新node和node.left的高度
        updateHeight(node);
        updateHeight(left);
        return left;
    }

    //更新根节点的高度
    private void updateHeight(Node node) {
        node.setHeight(Math.max(height(node.getLeft()), height(node.getRight())) + 1);
    }

    //先左旋后右旋
    private Node leftRightRotate(Node node) {
        node.setLeft(leftRotate(node.getLeft()));
        return rightRotate(node);
    }

    //先右旋后左旋
    private Node rightLeftRotate(Node node) {
        node.setRight(rightRotate(node.getRight()));
        return leftRotate(node);
    }


    public int height(Node node) {
        if (node == null) {
            return 0;
        } else {
            return node.getHeight();
        }

    }

    private boolean hasLeft(Node node) {
        return node.getLeft() != null;
    }


    private boolean hasRight(Node node) {
        return node.getLeft() != null || node.getRight() != null;
    }

    private boolean hasChild(Node node) {
        return node.getLeft() != null || node.getRight() != null;
    }

    private int getKey(Node node) {
        return node.getEntry().getKey();
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
    }
}
