package file;

/**
 * @author lilei
 **/
public class CriteriaBuilder {

    private Criteria eq(FileAttribute attribute, String value) {
        return new EqCriteria(attribute, value);
    }

}
