package tree;

//可编码
public interface Coded<K> {
    void listCode();
    byte[] encode(K[]k);
    Object[]decode(byte[]bytes);
}
