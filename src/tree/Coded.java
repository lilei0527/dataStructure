package tree;

//可编码
public interface Coded<K> {
    byte[] encode(K[]k);
    Object[]decode(byte[]bytes);
}
