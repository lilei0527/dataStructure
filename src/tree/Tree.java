package tree;

import java.util.List;

@SuppressWarnings("unused")
public interface Tree<K, V> {


    void insert(K k, V v);

    V search(K key);

    V delete(K key);

    void print();

    List<Entry<K, V>> bfsScan();//广度优先遍历

    List<Entry<K, V>> dfsScan(DFSType type);//深度度优先遍历

     enum DFSType{
        FRONT,MIDDLE,BEHIND
    }

}
