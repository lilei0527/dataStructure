package file;

import java.io.File;
import java.util.Date;

public class GreaterThanCriteria extends ComparableCriteria implements Visitor{
    public GreaterThanCriteria(FileAttribute attribute, Object object) {
        super(attribute, object);
    }

    @Override
    public boolean visitName(File file) {
        throw new IllegalArgumentException("fileName can not greaterThan");
    }

    @Override
    public boolean visitType(File file) {
        throw new IllegalArgumentException("fileType can not greaterThan");
    }

    @Override
    public boolean visitLength(File file) {
        return file.length() > (int)object;
    }

    @Override
    public boolean visitUpdateTime(File file) {
        return file.lastModified() > ((Date) object).getTime();    }
}
