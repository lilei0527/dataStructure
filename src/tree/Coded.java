package tree;

//可编码
public interface Coded {
    byte[] encode(byte[]bytes);
    byte[]decode(byte[]bytes);

}
