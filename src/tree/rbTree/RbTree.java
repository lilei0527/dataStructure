package tree.rbTree;

import tree.Entry;
import tree.Tree;



/**
 * @author lilei
 * create at 2020/2/29 15:35
 * <p>
 * 红黑树定义
 * 1.节点有颜色的属性,根节点必须为黑色
 * 2.所有叶子节点是null节点，且颜色为黑色
 * 3.如果一个节点是红色的那么它的两个节点都是黑色
 * 4.从任何一个节点到其叶子节点的简单路径都包含相同数目的黑色节点
 */
public class RbTree implements Tree {
    private static final boolean RED = false;
    private static final boolean BLACK = true;
    private Node root;

    @Override
    public String toString() {
        return "RbTree{" +
                "root=" + root +
                '}';
    }

    @Override
    public void insert(Entry entry) {
        if (root == null) {
            root = new Node(null, entry);
        } else {
            Node p = root;
            Node parent;
            //迭代
            do {
                parent = p;

                if (p.entry.key < entry.key) {
                    p = p.right;
                } else if (p.entry.key > entry.key) {
                    p = p.left;
                } else {
                    p.entry = entry;
                }
            } while (p != null);


            Node node = new Node(parent, entry);
            if (parent.entry.key < entry.key) {
                parent.right = node;
            } else {
                parent.left = node;
            }

            fixAfterInsertion(node);
        }


    }

    @Override
    public Entry search(int key) {
        return search(root, key).entry;
    }

    public Node search(Node node, int key) {
        if (node == null) {
            return null;
        } else {
            if (node.entry.key > key) {
                return search(node.left, key);
            }
            if (node.entry.key < key) {
                return search(node.right, key);
            }
            return node;
        }
    }

    @Override
    public Entry delete(int key) {
        Node node = search(root, key);
        if (node == null) {
            return null;
        }
        deleteNode(node);
        return node.entry;
    }

    public void deleteNode(Node p) {
        //待删除节点有两个孩子
        if (p.left != null && p.right != null) {
            Node s = getMin(p);
            //将待删除节点属性赋值到后继节点
            s.entry = p.entry;
            p = s;
        }

        Node replacement = p.left != null ? p.left : p.right;

        if (replacement != null) {
            if (p.parent == null) {
                root = replacement;
            } else if (p == p.parent.left) {
                p.parent.left = replacement;
            } else {
                p.parent.right = replacement;
            }
            replacement.parent = p.parent;

            if (p.color == BLACK) {
                fixAfterDelete(replacement);
            }
        } else if (p.parent == null) {
            root = null;
        } else {
            if (p.color == BLACK)
                fixAfterDelete(p);

            if (p.parent != null) {
                if (p == p.parent.left)
                    p.parent.left = null;
                else if (p == p.parent.right)
                    p.parent.right = null;
                p.parent = null;
            }
        }


    }

    private Node getMin(Node node) {
        Node p = null;
        while (node != null) {
            p = node;
            node = node.left;
        }
        return p;
    }

    /**
     * @author lilei
     *
     * 四种异常情况
     *
     * 第一种：
     *      b
     *     /\
     *    r  r
     *   /
     *  node
     *
     *
     *  第二种：
     *      b                b
     *     /                /
     *    r       或者     r
     *   /                 \
     *  node               node
     *
     *
     *  第三种：
     *      b
     *     /\
     *    r  r
     *       \
     *       node
     *
     *
     *  第四种：
     *      b                b
     *      \                \
     *       r       或者     r
     *        \              /
     *        node          node
     *
     */
    private void fixAfterInsertion(Node node) {
        node.color = RED;

        while (node != root && node.parent.color == RED) {

            if (parentOf(node) == leftOf(parentOf(parentOf(node)))) {
                Node y = rightOf(parentOf(parentOf(node)));
                if (colorOf(y) == RED) {
                    //对应第一种情况
                    setColor(parentOf(parentOf(node)), RED);
                    setColor(y, BLACK);
                    setColor(parentOf(node), BLACK);
                    node = parentOf(parentOf(node));
                } else {
                    //对应第二种情况
                    if (rightOf(parentOf(node)) == node) {
                        node = parentOf(node);
                        rotateLeft(node);
                    }
                    setColor(parentOf(node), BLACK);
                    setColor(parentOf(parentOf(node)), RED);
                    rotateRight(parentOf(parentOf(node)));
                }
            } else {
                Node y = leftOf(parentOf(parentOf(node)));
                if (colorOf(y) == RED) {
                    //对应第三种情况
                    setColor(parentOf(parentOf(node)), RED);
                    setColor(y, BLACK);
                    setColor(parentOf(node), BLACK);
                    node = parentOf(parentOf(node));
                } else {
                    //对应第四种情况
                    if (leftOf(parentOf(node)) == node) {
                        node = parentOf(node);
                        rotateRight(node);
                    }
                    setColor(parentOf(node), BLACK);
                    setColor(parentOf(parentOf(node)), RED);
                    rotateLeft(parentOf(parentOf(node)));
                }
            }
        }

        root.color = BLACK;
    }

    /**
     * @author lilei
     *
     */
    private void fixAfterDelete(Node x) {
        while (x != root && colorOf(x) == BLACK) {
            if (x == leftOf(parentOf(x))) {
                Node sib = rightOf(parentOf(x));

                if (colorOf(sib) == RED) {
                    setColor(sib, BLACK);
                    setColor(parentOf(x), RED);
                    rotateLeft(parentOf(x));
                    sib = rightOf(parentOf(x));
                }

                if (colorOf(leftOf(sib)) == BLACK &&
                        colorOf(rightOf(sib)) == BLACK) {
                    setColor(sib, RED);
                    x = parentOf(x);
                } else {
                    if (colorOf(rightOf(sib)) == BLACK) {
                        setColor(leftOf(sib), BLACK);
                        setColor(sib, RED);
                        rotateRight(sib);
                        sib = rightOf(parentOf(x));
                    }
                    setColor(sib, colorOf(parentOf(x)));
                    setColor(parentOf(x), BLACK);
                    setColor(rightOf(sib), BLACK);
                    rotateLeft(parentOf(x));
                    x = root;
                }
            } else { // symmetric
                Node sib = leftOf(parentOf(x));

                if (colorOf(sib) == RED) {
                    setColor(sib, BLACK);
                    setColor(parentOf(x), RED);
                    rotateRight(parentOf(x));
                    sib = leftOf(parentOf(x));
                }

                if (colorOf(rightOf(sib)) == BLACK &&
                        colorOf(leftOf(sib)) == BLACK) {
                    setColor(sib, RED);
                    x = parentOf(x);
                } else {
                    if (colorOf(leftOf(sib)) == BLACK) {
                        setColor(rightOf(sib), BLACK);
                        setColor(sib, RED);
                        rotateLeft(sib);
                        sib = leftOf(parentOf(x));
                    }
                    setColor(sib, colorOf(parentOf(x)));
                    setColor(parentOf(x), BLACK);
                    setColor(leftOf(sib), BLACK);
                    rotateRight(parentOf(x));
                    x = root;
                }
            }
        }

        setColor(x, BLACK);
    }

    private void rotateLeft(Node p) {
        if (p != null) {
            Node r = p.right;
            p.right = r.left;
            if (r.left != null)
                r.left.parent = p;
            r.parent = p.parent;
            if (p.parent == null)
                root = r;
            else if (p.parent.left == p)
                p.parent.left = r;
            else
                p.parent.right = r;
            r.left = p;
            p.parent = r;
        }
    }

    private void rotateRight(Node p) {
        if (p != null) {
            Node l = p.left;
            p.left = l.right;
            if (l.right != null) l.right.parent = p;
            l.parent = p.parent;
            if (p.parent == null)
                root = l;
            else if (p.parent.right == p)
                p.parent.right = l;
            else p.parent.left = l;
            l.right = p;
            p.parent = l;
        }
    }


    private static boolean colorOf(Node node) {
        return (node == null ? BLACK : node.color);
    }

    private static Node parentOf(Node node) {
        return (node == null ? null : node.parent);
    }

    private static void setColor(Node node, boolean c) {
        if (node != null)
            node.color = c;
    }

    private static Node leftOf(Node node) {
        return (node == null) ? null : node.left;
    }

    private static Node rightOf(Node node) {
        return (node == null) ? null : node.right;
    }

    public static void main(String[] args) {
        RbTree rbTree = new RbTree();
        rbTree.insert(new Entry(2, 2));
        rbTree.insert(new Entry(21, 22));
        rbTree.insert(new Entry(4, 4));
        rbTree.insert(new Entry(1, 41));
        rbTree.insert(new Entry(3, 3));
        rbTree.insert(new Entry(23, 23));
        System.out.println(rbTree.root);
    }
}
