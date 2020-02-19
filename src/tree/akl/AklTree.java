package tree.akl;

import tree.Entry;
import tree.Tree;

@SuppressWarnings("unused")
public class AklTree implements Tree {
    private Node root;

    @Override
    public void insert(Entry entry) {
        if (!hasRoot()) {
            root = createNode(entry);
        } else {
            insert(entry, root);
        }
    }

    public void insert(Entry entry, Node node) {
        Node newNode = createNode(entry);
        if (isLeft(entry, node)) {
            if (hasLeft(node)) {
                insert(entry, node.getLeft());
            } else {
                node.setLeft(newNode);
            }
        }

        if (isRight(entry, node)) {
            if (hasRight(node)) {
                insert(entry, node.getRight());
            } else {
                node.setRight(newNode);
            }
        }

        if (isSame(entry, node)) {
            if (!hasLeft(node)) {
                node.setLeft(newNode);
            } else {
                if (!hasRight(node)) {
                    node.setRight(newNode);
                } else {
                    //左右节点都存在的情况,往左边插入
                    insert(entry, node.getLeft());
                }
            }
        }
    }

    @Override
    public void search(Entry key) {

    }


    @Override
    public boolean delete(Entry key) {
        return false;
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
        return entry.getKey() < getKey(node);
    }

    private boolean isRight(Entry entry, Node node) {
        return entry.getKey() > getKey(node);
    }

    private boolean isSame(Entry entry, Node node) {
        return entry.getKey() == getKey(node);
    }

    private Node createNode(Entry entry) {
        return new Node(entry, null, null);
    }

    private boolean isRoot(Node node) {
        return node == root;
    }


    public static void main(String[] args) {
        AklTree tree = new AklTree();
        tree.insert(new Entry(1, 1));
        tree.insert(new Entry(4, 4));
        tree.insert(new Entry(8, 8));
        tree.insert(new Entry(3, 3));
        tree.insert(new Entry(3, 3));
        tree.insert(new Entry(2, 2));
        System.out.println(tree.root);
    }
}
