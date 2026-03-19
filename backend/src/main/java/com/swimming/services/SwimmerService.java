
Public SwimmerService (SwimmerRepository swimmerRepository) 
{
    {
        this.swimmerRepository = swimmerRepository;
    }

    public List<Swimmer> getAllSwimmers() {
        return swimmerRepository.findAll();
    }

    public Swimmer getSwimmerById(Long id) {
        return swimmerRepository.findById(id).orElse(null);
    }

    public Swimmer createSwimmer(Swimmer swimmer) {
        return swimmerRepository.save(swimmer);
    }

    public Swimmer updateSwimmer(Long id, Swimmer swimmerDetails) {
        Swimmer swimmer = swimmerRepository.findById(id).orElse(null);
        if (swimmer != null) {
            swimmer.setName(swimmerDetails.getName());
            swimmer.setAge(swimmerDetails.getAge());
            swimmer.setGender(swimmerDetails.getGender());
            return swimmerRepository.save(swimmer);
        }
        return null;
    }

    public void deleteSwimmer(Long id) {
        swimmerRepository.deleteById(id);
    }
}