package com.swimming.controllers;

import java.util.List;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.swimming.models.Swimmer;
import com.swimming.services.SwimmerService;

@RestController
@RequestMapping("/swimmer")

public class SwimmerController {
    
private SwimmerService service ;

public SwimmerController (SwimmerService service)
{
    this.service=service;
}

@GetMapping
public List<Swimmer> findAllSwimmers ()
{
    return service.getAll();
}

@GetMapping("/{id}")
public Swimmer FindSwimmer (@PathVariable Long id)
{
    return service.getSwimmerById(id);
}

@PostMapping 
public Swimmer AddSwimmer (@RequestBody Swimmer swimmer)
{
    return service.CreateSwimmer(swimmer);
}

@PutMapping ("/{Id}")
public Swimmer UpdateSwimmer (@PathVariable Long Id,@RequestBody Swimmer swimmer)
{
    swimmer.setId(Id);

    return service.saveSwimmer(swimmer);
}

@DeleteMapping("/{Id}")
public void DeleteSwimmer(@PathVariable Long Id)
{
 service.deleteSwimmer(Id);

}


}