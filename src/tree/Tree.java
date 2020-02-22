package tree;

public interface Tree {
    void insert(Entry key);

    Entry search(int key);

    boolean delete(int key);

}
