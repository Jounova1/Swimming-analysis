package com.swimming.models;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;

@Entity
public class Swimmer
{
@Id
@GeneratedValue
 
private Long id;
private String name;
private String age;
private String phone_num;
private String team;
private String gender ;
public Object getGender() {
    // TODO Auto-generated method stub
    throw new UnsupportedOperationException("Unimplemented method 'getGender'");
}
public void setGender(Object gender2) {
    // TODO Auto-generated method stub
    throw new UnsupportedOperationException("Unimplemented method 'setGender'");
}
public Object getAge() {
    // TODO Auto-generated method stub
    throw new UnsupportedOperationException("Unimplemented method 'getAge'");
}
public Object getName() {
    // TODO Auto-generated method stub
    throw new UnsupportedOperationException("Unimplemented method 'getName'");
}
public void setAge(Object age2) {
    // TODO Auto-generated method stub
    throw new UnsupportedOperationException("Unimplemented method 'setAge'");
}
public void setName(Object name2) {
    // TODO Auto-generated method stub
    throw new UnsupportedOperationException("Unimplemented method 'setName'");
}
}