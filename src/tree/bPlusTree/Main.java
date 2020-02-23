package tree.bPlusTree;

import tree.Entry;

public class Main {

    public static void main(String[] args) {
        BPlusTree btree = new BPlusTree(4 );
        Entry entry = new Entry(1,"123");
        Entry entry1 = new Entry(2,"123");
        Entry entry2 = new Entry(3,"123");
        Entry entry3 = new Entry(4,"123");
        Entry entry4 = new Entry(5,"123");
        Entry entry5 = new Entry(6,"123");
        Entry entry6 = new Entry(7,"12300");
        Entry entry7 = new Entry(8,"546");
        Entry entry8 = new Entry(9,"123");
        Entry entry9 = new Entry(10,"123");
        Entry entry10 = new Entry(11,"123");
        Entry entry11 = new Entry(12,"123");
        Entry entry12 = new Entry(13,"123");
        Entry entry14 = new Entry(15,"12345");
        Entry entry15 = new Entry(16,"12345");
        Entry entry16 = new Entry(17,"12345");
        Entry entry17 = new Entry(18,"12345");
        Entry entry18 = new Entry(19,"12345");
        Entry entry19 = new Entry(20,"12345");
        Entry entry20 = new Entry(21,"12345");

        btree.insert(entry);
        btree.insert(entry5);
        btree.insert(entry9);
        btree.insert(entry1);
        btree.insert(entry7);
        btree.insert(entry10);
        btree.insert(entry17);
        btree.insert(entry2);
        btree.insert(entry14);
        btree.insert(entry16);
        btree.insert(entry11);
        btree.insert(entry12);
        btree.insert(entry3);
        btree.insert(entry8);
        btree.insert(entry18);
        btree.insert(entry15);
        btree.insert(entry4);
        btree.insert(entry19);
        btree.insert(entry6);
        btree.insert(entry20);


        btree.printBtree(btree.getRoot());

        btree.delete(1);
        btree.printBtree(btree.getRoot());

        btree.delete(0);
        btree.printBtree(btree.getRoot());

        btree.delete(2);
        btree.printBtree(btree.getRoot());

        btree.delete(11);
        btree.printBtree(btree.getRoot());

        btree.delete(3);
        btree.printBtree(btree.getRoot());

        btree.delete(4);
        btree.printBtree(btree.getRoot());

        btree.delete(5);
        btree.printBtree(btree.getRoot());

        btree.delete(9);
        btree.printBtree(btree.getRoot());

        btree.delete(6);
        btree.printBtree(btree.getRoot());

        btree.delete(13);
        btree.printBtree(btree.getRoot());

        btree.delete(7);
        btree.printBtree(btree.getRoot());

        btree.delete(10);
        btree.printBtree(btree.getRoot());

        btree.delete(18);
        btree.printBtree(btree.getRoot());

        btree.delete(8);
        btree.printBtree(btree.getRoot());

        btree.delete(12);
        btree.printBtree(btree.getRoot());

        btree.delete(20);
        btree.printBtree(btree.getRoot());

        btree.delete(19);
        btree.printBtree(btree.getRoot());

        btree.delete(15);
        btree.printBtree(btree.getRoot());

        btree.delete(17);
        btree.printBtree(btree.getRoot());



        System.out.println(btree.search(18,btree.getRoot(), BPlusTree.ENTRY));

    }
}
