package file;

import java.io.File;
import java.util.Date;

public class LessThanOrEqualCriteria extends ComparableCriteria implements Visitor {
    public LessThanOrEqualCriteria(FileAttribute attribute, Object object) {
        super(attribute, object);
    }

    @Override
    public boolean visitName(File file) {
        throw new IllegalArgumentException("fileName can not lessThanOrEqual");
    }

    @Override
    public boolean visitType(File file) {
        throw new IllegalArgumentException("fileType can not lessThanOrEqual");
    }

    @Override
    public boolean visitLength(File file) {
        return file.length() <= (int)object;
    }

    @Override
    public boolean visitUpdateTime(File file) {
        return file.lastModified() <= ((Date) object).getTime();    }
}
