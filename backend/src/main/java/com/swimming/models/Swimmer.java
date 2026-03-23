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
<<<<<<< HEAD

public Long getId() {
    return id;
}

public void setId(Long id) {
    this.id = id;
}

public String getName() {
    return name;
}

public void setName(String name) {
    this.name = name;
}

public String getAge() {
    return age;
}

public void setAge(String age) {
    this.age = age;
}

public String getPhone_num() {
    return phone_num;
}

public void setPhone_num(String phone_num) {
    this.phone_num = phone_num;
}

public String getTeam() {
    return team;
}

public void setTeam(String team) {
    this.team = team;
}



=======
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
>>>>>>> 25d448ca2a57475841eb07f34c87fca9cb1020d1
}