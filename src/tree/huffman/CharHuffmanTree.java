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

    public Character[] toArray() {
        char[] chars = content.toCharArray();
        Character[] characters = new Character[chars.length];
        int index = 0;
        for (char c : chars) {
            characters[index++] = c;
        }
        return characters;
    }

    @Override
    public byte getBinCode(Character character) {
        return (byte) character.charValue();
    }
}
