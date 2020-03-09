package tree;

import tree.akl.AvlTree;
import tree.bPlusTree.BPlusTree;
import tree.rbTree.RbTree;

public class TreeTest {
    public static void main(String[] args) {

        System.out.println("红黑树:");
        RbTree<Integer, Integer> rbTree = new RbTree<>();
        rbTree.insert(1, 1);
        rbTree.insert(3, 3);
        rbTree.insert(2, 2);
        rbTree.insert(5, 5);
        rbTree.insert(77, 77);
        rbTree.insert(32, 32);
        rbTree.insert(11, 11);
        rbTree.insert(12, 112);
        rbTree.insert(13, 561);
        rbTree.insert(14, 561);
        rbTree.print();

        System.out.println("AVL树:");
        AvlTree<Integer, Integer> avlTree = new AvlTree<>();
        avlTree.insert(1, 1);
        avlTree.insert(3, 3);
        avlTree.insert(2, 2);
        avlTree.insert(5, 5);
        avlTree.insert(77, 77);
        avlTree.insert(32, 32);
        avlTree.insert(11, 11);
        avlTree.insert(12, 112);
        avlTree.insert(13, 561);
        avlTree.insert(14, 561);
        avlTree.print();

        System.out.println("B+树:");
        BPlusTree<Integer, Integer> bPlusTree = new BPlusTree<>(4);
        bPlusTree.insert(1, 1);
        bPlusTree.insert(3, 3);
        bPlusTree.insert(2, 2);
        bPlusTree.insert(5, 5);
        bPlusTree.insert(77, 77);
        bPlusTree.insert(32, 32);
        bPlusTree.insert(11, 11);
        bPlusTree.insert(12, 112);
        bPlusTree.insert(13, 561);
        bPlusTree.insert(14, 561);
        bPlusTree.print();


    }
}
