const authService = {
  login: async (credentials) => {
    return { success: true, data: credentials };
  },
  logout: async () => {
    return { success: true };
  },
};

export default authService;
