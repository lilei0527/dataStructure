package file;

import java.io.File;
import java.util.Date;

public class NotEqualCriteria extends ComparableCriteria implements Visitor{
    public NotEqualCriteria(FileAttribute attribute, Object object) {
        super(attribute, object);
    }

    @Override
    public boolean visitName(File file) {
        return !file.getName().equals(object);
    }

    @Override
    public boolean visitType(File file) {
        return !getSuffix(file).equals(object);
    }

    @Override
    public boolean visitLength(File file) {
        return !(file.length() == (int) object);
    }

    @Override
    public boolean visitUpdateTime(File file) {
        return !(file.lastModified() == ((Date) object).getTime());
    }
}
