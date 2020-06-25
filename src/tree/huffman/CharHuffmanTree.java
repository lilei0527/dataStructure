package tree.huffman;


/**
 * @author lilei
 **/
public class CharHuffmanTree extends HuffmanTree<Character> {
    private String content;

    public CharHuffmanTree( String content) {
        super(content.length());
        this.content = content;
        addNode();
        create();
    }

    public byte[] toArray() {
        return content.getBytes();
    }
}
