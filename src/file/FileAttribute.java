package file;

public enum FileAttribute {

    NAME("name", "文件名"),
    TYPE("type", "文件类型"),
    SIZE("size", "文件大小"),
    UPDATE_TIME("updateTime", "更新时间");
    private String name;
    private String desc;

    FileAttribute(String name, String desc) {
        this.name = name;
        this.desc = desc;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDesc() {
        return desc;
    }

    public void setDesc(String desc) {
        this.desc = desc;
    }
}
