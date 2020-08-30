package file;


import java.io.File;

/**
 * @author lilei
 **/
public abstract class ComparableCriteria extends AbstractSimpleCriteria {
    protected Object object;

    public ComparableCriteria(FileAttribute attribute, Object object) {
        super(attribute);
        this.object = object;
    }

    protected String getSuffix(File file) {
        String name = file.getName();
        int i = name.lastIndexOf('.');
        if (i == -1) {
            return "";
        }
        return name.substring(i+1);
    }


}
