package tree.btree;

public class Main {

    public static void main(String[] args) {
        Btree btree = new Btree(4 );
        KeyAndValue keyAndValue = new KeyAndValue(1,"123");
        KeyAndValue keyAndValue1 = new KeyAndValue(2,"123");
        KeyAndValue keyAndValue2 = new KeyAndValue(3,"123");
        KeyAndValue keyAndValue3 = new KeyAndValue(4,"123");
        KeyAndValue keyAndValue4 = new KeyAndValue(5,"123");
        KeyAndValue keyAndValue5 = new KeyAndValue(6,"123");
        KeyAndValue keyAndValue6 = new KeyAndValue(7,"12300");
        KeyAndValue keyAndValue7 = new KeyAndValue(8,"546");
        KeyAndValue keyAndValue8 = new KeyAndValue(9,"123");
        KeyAndValue keyAndValue9 = new KeyAndValue(10,"123");
        KeyAndValue keyAndValue10 = new KeyAndValue(11,"123");
        KeyAndValue keyAndValue11 = new KeyAndValue(12,"123");
        KeyAndValue keyAndValue12 = new KeyAndValue(13,"123");
        KeyAndValue keyAndValue14 = new KeyAndValue(15,"12345");
        KeyAndValue keyAndValue15 = new KeyAndValue(16,"12345");
        KeyAndValue keyAndValue16 = new KeyAndValue(17,"12345");
        KeyAndValue keyAndValue17 = new KeyAndValue(18,"12345");
        KeyAndValue keyAndValue18 = new KeyAndValue(19,"12345");
        KeyAndValue keyAndValue19 = new KeyAndValue(20,"12345");
        KeyAndValue keyAndValue20 = new KeyAndValue(21,"12345");

        btree.insert(keyAndValue);
        btree.insert(keyAndValue5);
        btree.insert(keyAndValue9);
        btree.insert(keyAndValue1);
        btree.insert(keyAndValue7);
        btree.insert(keyAndValue10);
        btree.insert(keyAndValue17);
        btree.insert(keyAndValue2);
        btree.insert(keyAndValue14);
        btree.insert(keyAndValue16);
        btree.insert(keyAndValue11);
        btree.insert(keyAndValue12);
        btree.insert(keyAndValue3);
        btree.insert(keyAndValue8);
        btree.insert(keyAndValue18);
        btree.insert(keyAndValue15);
        btree.insert(keyAndValue4);
        btree.insert(keyAndValue19);
        btree.insert(keyAndValue6);
        btree.insert(keyAndValue20);


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


        System.out.println(btree.search(18,btree.getRoot(),Btree.INT));

    }
}
