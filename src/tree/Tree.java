package tree;

public interface Tree {
    void insert(Entry key);

    Entry search(int key);

    Entry delete(int key);

}
