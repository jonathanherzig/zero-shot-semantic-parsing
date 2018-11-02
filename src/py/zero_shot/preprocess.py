import delex_data
import create_dev_split
import prepare_cross_domain

if __name__ == "__main__":
    domains = ['publications', 'restaurants', 'housing', 'recipes', 'socialnetwork', 'calendar', 'blocks']

    print "\nDelexicalize data"
    delex_data.delex_domains(domains)

    print "\nCreate dev splits"
    create_dev_split.create_dev_split()

    print "\nPrepare cross domain training data"
    prepare_cross_domain.process_cross_domain(domains, is_split_test=False)
    prepare_cross_domain.process_cross_domain(domains, is_split_test=True)

